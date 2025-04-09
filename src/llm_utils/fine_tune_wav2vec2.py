import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
from typing import Dict, List, Union
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fine_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RainSoundDataset:
    """
    Custom dataset class for rain sound intensity classification.
    This will be used to prepare our audio data for fine-tuning.
    """
    def __init__(self, audio_paths: List[str], labels: List[int], processor: Wav2Vec2Processor):
        """
        Initialize the dataset.
        
        Args:
            audio_paths: List of paths to audio files
            labels: List of corresponding labels
            processor: Wav2Vec2Processor for audio preprocessing
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single audio file.
        
        Args:
            idx: Index of the audio file to process
            
        Returns:
            Dictionary containing processed audio and label
        """
        try:
            # Load audio file
            audio_path = self.audio_paths[idx]
            audio = Audio().decode_example(Audio().encode_example(audio_path))
            
            # Process audio
            input_values = self.processor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            ).input_values[0]
            
            return {
                "input_values": input_values,
                "labels": torch.tensor(self.labels[idx])
            }
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {str(e)}")
            raise

def prepare_dataset(audio_dir: str, val_split: float = 0.2) -> Dict[str, Dict[str, List[str]]]:
    """
    Prepare the dataset by collecting audio files and their labels, with validation split.
    Specifically focused on rain intensity classification.
    
    Args:
        audio_dir: Directory containing audio files
        val_split: Proportion of data to use for validation (default: 0.2 or 20%)
        
    Returns:
        Dictionary containing training and validation datasets
    """
    audio_paths = []
    labels = []
    
    # Map rain intensities to labels
    label_map = {
        "light": 0,
        "moderate": 1,
        "heavy": 2
    }
    
    # Walk through the directory and collect audio files
    for intensity_dir in ["light", "moderate", "heavy"]:
        intensity_path = os.path.join(audio_dir, intensity_dir)
        if not os.path.exists(intensity_path):
            logger.warning(f"Directory not found: {intensity_path}")
            continue
            
        for file in os.listdir(intensity_path):
            if file.endswith(('.wav', '.mp3')):
                audio_paths.append(os.path.join(intensity_path, file))
                labels.append(label_map[intensity_dir])
                logger.debug(f"Added {file} as {intensity_dir} rain")
    
    if not audio_paths:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    # Shuffle the data
    indices = np.random.permutation(len(audio_paths))
    audio_paths = [audio_paths[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Split into train and validation
    split_idx = int(len(audio_paths) * (1 - val_split))
    
    return {
        "train": {
            "audio_paths": audio_paths[:split_idx],
            "labels": labels[:split_idx]
        },
        "validation": {
            "audio_paths": audio_paths[split_idx:],
            "labels": labels[split_idx:]
        }
    }

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Tuple containing predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('logs/confusion_matrix.png')
    plt.close()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize processor and model
    logger.info("Loading Wav2Vec2 model and processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=3,  # Number of rain intensity classes
        label2id={"light": 0, "moderate": 1, "heavy": 2},
        id2label={0: "light", 1: "moderate", 2: "heavy"}
    )
    model.to(device)
    
    # Prepare dataset with validation split
    logger.info("Preparing dataset...")
    dataset = prepare_dataset("data/processed/rain_sounds", val_split=0.2)
    
    # Create training and validation datasets
    train_dataset = RainSoundDataset(
        dataset["train"]["audio_paths"],
        dataset["train"]["labels"],
        processor
    )
    
    val_dataset = RainSoundDataset(
        dataset["validation"]["audio_paths"],
        dataset["validation"]["labels"],
        processor
    )
    
    # Log dataset sizes and class distribution
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Count samples per class
    train_labels = dataset["train"]["labels"]
    val_labels = dataset["validation"]["labels"]
    
    logger.info("Training class distribution:")
    logger.info(f"Light rain: {train_labels.count(0)} samples")
    logger.info(f"Moderate rain: {train_labels.count(1)} samples")
    logger.info(f"Heavy rain: {train_labels.count(2)} samples")
    
    logger.info("Validation class distribution:")
    logger.info(f"Light rain: {val_labels.count(0)} samples")
    logger.info(f"Moderate rain: {val_labels.count(1)} samples")
    logger.info(f"Heavy rain: {val_labels.count(2)} samples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/fine_tuned_wav2vec2_rain_intensity",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=500,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir="logs",
        report_to="tensorboard",
        save_total_limit=2,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    
    # Initialize trainer with validation dataset
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model("models/fine_tuned_wav2vec2_rain_intensity/final")
    processor.save_pretrained("models/fine_tuned_wav2vec2_rain_intensity/final")
    
    # Log final metrics
    logger.info("Training completed! Final metrics:")
    final_metrics = trainer.evaluate()
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value}")
    
    logger.info("Model saved and training completed!")

if __name__ == "__main__":
    main() 