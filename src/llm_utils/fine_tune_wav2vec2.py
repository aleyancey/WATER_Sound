import os
import torch
import logging
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForAudioClassification:
    """
    Data collator for audio classification.
    Handles batching of audio features and labels.
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.feature_extractor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Add labels
        if "surface_type" in features[0].keys():
            label_name = "surface_type"
            labels = [feature[label_name] for feature in features]
            batch["labels"] = torch.tensor(labels)
            
        return batch

def prepare_dataset(dataset_path: str = "data/processed"):
    """
    Prepare dataset for training.
    
    Args:
        dataset_path: Path to the processed dataset
        
    Returns:
        train_dataset, eval_dataset
    """
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Create label mapping
    unique_labels = sorted(set(dataset["train"]["surface_type"]))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    logger.info(f"Found {len(unique_labels)} unique surface types: {unique_labels}")
    
    # Convert labels to ids
    def convert_labels(example):
        example["surface_type"] = label2id[example["surface_type"]]
        return example
    
    train_dataset = dataset["train"].map(convert_labels)
    eval_dataset = dataset["test"].map(convert_labels)
    
    return train_dataset, eval_dataset, label2id, id2label

def train_model(
    train_dataset,
    eval_dataset,
    label2id,
    id2label,
    model_name: str = "facebook/wav2vec2-base",
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    output_dir: str = "models/wav2vec2-rain"
):
    """
    Fine-tune Wav2Vec2 model for audio classification.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        label2id: Label to ID mapping
        id2label: ID to label mapping
        model_name: Pre-trained model name
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save model
    """
    try:
        # Load pre-trained model and feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
        )
        
        # Create data collator
        data_collator = DataCollatorForAudioClassification(
            feature_extractor=feature_extractor,
            padding=True
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            do_eval=True,
            eval_steps=100,
            save_steps=100,
            logging_steps=10,
            save_total_limit=2,
            gradient_accumulation_steps=2,
            logging_dir=f"{output_dir}/logs",
            report_to="tensorboard",
            remove_unused_columns=False,
            dataloader_num_workers=2,
            use_mps_device=True  # Enable Apple Silicon GPU support
        )
        
        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        feature_extractor.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        # Evaluate model
        metrics = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def compute_metrics(pred):
    """
    Compute metrics for evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

def main():
    """Main entry point for fine-tuning."""
    try:
        # Prepare dataset
        train_dataset, eval_dataset, label2id, id2label = prepare_dataset()
        
        # Create output directory
        output_dir = "models/wav2vec2-rain"
        os.makedirs(output_dir, exist_ok=True)
        
        # Train model
        train_model(
            train_dataset,
            eval_dataset,
            label2id,
            id2label,
            output_dir=output_dir
        )
        
        logger.info("Fine-tuning completed successfully")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 