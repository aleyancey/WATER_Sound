import os
import torch
import logging
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioClassifier(nn.Module):
    """
    Custom neural network for audio classification using YAMNET features.
    Includes both classification and regression heads for different audio characteristics.
    """
    def __init__(self, yamnet_dim=1024, num_classes=10):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(yamnet_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head for surface type
        self.surface_classifier = nn.Linear(256, num_classes)
        
        # Regression heads for audio characteristics
        self.bpm_regressor = nn.Linear(256, 1)
        self.loudness_regressor = nn.Linear(256, 1)
        self.brightness_regressor = nn.Linear(256, 1)
        self.noisiness_regressor = nn.Linear(256, 1)

    def forward(self, x):
        # Shared features
        shared_features = self.shared(x)
        
        # Multiple outputs
        return {
            'surface_type': self.surface_classifier(shared_features),
            'bpm': self.bpm_regressor(shared_features),
            'loudness': self.loudness_regressor(shared_features),
            'brightness': self.brightness_regressor(shared_features),
            'noisiness': self.noisiness_regressor(shared_features)
        }

class AudioDataset(torch.utils.data.Dataset):
    """Dataset class for audio features."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(
    train_dataset,
    eval_dataset,
    label2id,
    id2label,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    output_dir: str = "models/audio_classifier"
):
    """
    Train the audio classifier model.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        label2id: Label to ID mapping
        id2label: ID to label mapping
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save model
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        train_features = np.stack(train_dataset['yamnet_embedding'])
        train_labels = np.array([label2id[label] for label in train_dataset['surface_type']])
        eval_features = np.stack(eval_dataset['yamnet_embedding'])
        eval_labels = np.array([label2id[label] for label in eval_dataset['surface_type']])
        
        # Create data loaders
        train_data = AudioDataset(train_features, train_labels)
        eval_data = AudioDataset(eval_features, eval_labels)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_data, batch_size=batch_size)
        
        # Initialize model
        model = AudioClassifier(num_classes=len(label2id))
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        
        # Loss functions
        classification_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                outputs = model(batch_features)
                loss = classification_criterion(outputs['surface_type'], batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluate
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_features, batch_labels in eval_loader:
                    batch_features = batch_features.to(device)
                    outputs = model(batch_features)
                    preds = outputs['surface_type'].argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_labels.numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted'
            )
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Loss: {total_loss/len(train_loader):.4f}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'label2id': label2id,
                    'id2label': id2label,
                    'accuracy': accuracy,
                    'f1': f1
                }, f"{output_dir}/best_model.pt")
            
            scheduler.step()
        
        logger.info("Training completed!")
        logger.info(f"Best accuracy: {best_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def main():
    """Main entry point for training."""
    try:
        # Load dataset
        dataset = load_from_disk("data/processed")
        
        # Get label mappings
        unique_labels = sorted(set(dataset["train"]["surface_type"]))
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for label, i in label2id.items()}
        
        # Train model
        train_model(
            dataset["train"],
            dataset["test"],
            label2id,
            id2label,
            output_dir="models/audio_classifier"
        )
        
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise

if __name__ == "__main__":
    main() 