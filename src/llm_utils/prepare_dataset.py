import os
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2FeatureExtractor
from typing import Dict, List, Tuple

class RainSoundDataset:
    """
    Class to prepare and process rain sound dataset for fine-tuning.
    Handles audio processing, feature extraction, and dataset creation.
    """
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw/rain_sounds",
                 processed_dir: str = "data/processed",
                 sample_rate: int = 16000,
                 max_duration: float = 30.0):
        """
        Initialize the dataset processor.
        
        Args:
            raw_data_dir: Directory containing raw audio files
            processed_dir: Directory to save processed data
            sample_rate: Target sample rate for audio processing
            max_duration: Maximum duration of audio clips in seconds
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
    def load_metadata(self) -> Dict:
        """Load all metadata files and combine them into a single dictionary."""
        metadata = {}
        metadata_dir = self.raw_data_dir / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            surface_type = metadata_file.stem.replace("_metadata", "")
            with open(metadata_file, "r") as f:
                metadata[surface_type] = json.load(f)
        
        return metadata
    
    def process_audio(self, audio_path: Path) -> Tuple[np.ndarray, float]:
        """
        Process a single audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (processed_audio, duration)
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        
        # If audio is longer than max_duration, split into chunks
        duration = len(audio) / sr
        if duration > self.max_duration:
            # Take the middle portion
            start = int((len(audio) - self.max_duration * sr) / 2)
            end = start + int(self.max_duration * sr)
            audio = audio[start:end]
            duration = self.max_duration
        
        return audio, duration
    
    def extract_features(self, audio: np.ndarray) -> Dict:
        """
        Extract features from audio using Wav2Vec2 feature extractor.
        
        Args:
            audio: Processed audio array
            
        Returns:
            Dictionary containing input_values and attention_mask
        """
        # Extract features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return {
            "input_values": inputs.input_values[0],
            "attention_mask": inputs.attention_mask[0]
        }
    
    def create_dataset(self) -> DatasetDict:
        """
        Create a Hugging Face dataset from processed audio files.
        
        Returns:
            DatasetDict containing train and validation splits
        """
        metadata = self.load_metadata()
        examples = []
        
        # Process each surface type
        for surface_type, sounds in metadata.items():
            surface_dir = self.raw_data_dir / surface_type
            
            for sound in sounds:
                audio_path = surface_dir / f"{sound['name']}.mp3"
                
                if not audio_path.exists():
                    continue
                
                try:
                    # Process audio
                    audio, duration = self.process_audio(audio_path)
                    
                    # Extract features
                    features = self.extract_features(audio)
                    
                    # Create example
                    example = {
                        **features,
                        "surface_type": surface_type,
                        "duration": duration,
                        "original_name": sound['name'],
                        "tags": sound['tags'],
                        "description": sound['description']
                    }
                    
                    examples.append(example)
                    
                except Exception as e:
                    print(f"Error processing {audio_path}: {str(e)}")
                    continue
        
        # Create dataset
        dataset = Dataset.from_list(examples)
        
        # Split into train and validation
        dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Save dataset
        dataset_dict.save_to_disk(self.processed_dir)
        
        return dataset_dict

def main():
    # Initialize dataset processor
    processor = RainSoundDataset()
    
    # Create dataset
    print("Creating dataset...")
    dataset_dict = processor.create_dataset()
    
    print("\nDataset created successfully!")
    print(f"Train set size: {len(dataset_dict['train'])}")
    print(f"Validation set size: {len(dataset_dict['test'])}")
    print(f"\nDataset saved to: {processor.processed_dir}")

if __name__ == "__main__":
    main() 