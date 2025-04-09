import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    def __init__(self, data_dir, label_map):
        """Initialize the dataset with processed audio features.
        
        Args:
            data_dir (str): Directory containing processed feature files
            label_map (dict): Mapping from surface types to integer labels
        """
        self.data_dir = data_dir
        self.label_map = label_map
        self.files = []
        
        # Load all feature files
        for surface_type in label_map.keys():
            surface_dir = os.path.join(data_dir, surface_type)
            if os.path.exists(surface_dir):
                for file in os.listdir(surface_dir):
                    if file.endswith('.npz'):
                        self.files.append((os.path.join(surface_dir, file), label_map[surface_type]))
        
        logger.info(f"Loaded {len(self.files)} training files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        data = np.load(file_path)
        
        # Load normalized features
        mel_spec = torch.FloatTensor(data['mel_spectrogram'])
        mfcc = torch.FloatTensor(data['mfcc'])
        audio_features = torch.FloatTensor([
            data['tempo'],
            data['loudness'],
            data['brightness'],
            data['noisiness']
        ])
        
        return {
            'mel_spec': mel_spec,
            'mfcc': mfcc,
            'audio_features': audio_features,
            'label': label
        }

class AudioClassifier(nn.Module):
    def __init__(self, num_classes, num_audio_features=4):
        """Initialize the audio classifier model.
        
        Args:
            num_classes (int): Number of surface type classes
            num_audio_features (int): Number of audio characteristics
        """
        super(AudioClassifier, self).__init__()
        
        # Mel spectrogram processing
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # MFCC processing
        self.mfcc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Audio features processing
        self.audio_fc = nn.Sequential(
            nn.Linear(num_audio_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined features processing
        self.combined_fc = nn.Sequential(
            nn.Linear(128 * 16 * 16 + 64 * 8 * 8 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel_spec, mfcc, audio_features):
        # Process mel spectrogram
        mel_out = self.mel_conv(mel_spec)
        mel_out = mel_out.view(mel_out.size(0), -1)
        
        # Process MFCCs
        mfcc_out = self.mfcc_conv(mfcc)
        mfcc_out = mfcc_out.view(mfcc_out.size(0), -1)
        
        # Process audio features
        audio_out = self.audio_fc(audio_features)
        
        # Combine features
        combined = torch.cat([mel_out, mfcc_out, audio_out], dim=1)
        output = self.combined_fc(combined)
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the audio classifier model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    early_stopping_patience = 10
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            mel_spec = batch['mel_spec'].to(device)
            mfcc = batch['mfcc'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(mel_spec, mfcc, audio_features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                mel_spec = batch['mel_spec'].to(device)
                mfcc = batch['mfcc'].to(device)
                audio_features = batch['audio_features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(mel_spec, mfcc, audio_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                logger.info('Early stopping triggered')
                break

def main():
    # Load label mapping
    with open('data/processed/label_map.json', 'r') as f:
        label_map = json.load(f)
    
    # Create datasets
    train_dataset = AudioDataset('data/processed', label_map)
    val_dataset = AudioDataset('data/processed', label_map)  # In practice, split train/val
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model
    model = AudioClassifier(num_classes=len(label_map))
    
    # Train model
    train_model(model, train_loader, val_loader)

if __name__ == '__main__':
    main() 