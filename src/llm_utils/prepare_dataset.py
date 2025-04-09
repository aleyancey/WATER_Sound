import os
import json
import logging
import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import traceback
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RainSoundDataset:
    def __init__(self, raw_dir="data/raw/rain_sounds", processed_dir="data/processed"):
        """Initialize the dataset processor with directories for raw and processed data.
        
        Args:
            raw_dir (str): Directory containing raw audio files organized by surface type
            processed_dir (str): Directory to save processed features and metadata
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize audio transforms
        self.sample_rate = 16000  # Target sample rate
        self.n_mels = 128  # Number of mel bands
        self.n_fft = 1024  # FFT window size
        self.hop_length = 512  # Hop length for feature extraction
        self.max_length = 1000  # Maximum length for feature sequences
        
        # Initialize feature extractors
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            melkwargs={
                "n_fft": self.n_fft,
                "n_mels": self.n_mels,
                "hop_length": self.hop_length
            }
        )
        
        # Initialize scalers for feature normalization
        self.mel_scaler = StandardScaler()
        self.mfcc_scaler = StandardScaler()
        self.audio_scaler = StandardScaler()
        
        # Load or initialize processing progress
        self._load_progress()

    def _load_progress(self):
        """Load processing progress from JSON file."""
        progress_file = self.processed_dir / "processing_progress.json"
        if progress_file.exists():
            with open(progress_file) as f:
                self.progress = json.load(f)
            logger.info(f"Loaded processing progress: {len(self.progress['processed'])} files processed")
        else:
            self.progress = {"processed": [], "failed": []}

    def _save_progress(self):
        """Save processing progress to JSON file."""
        with open(self.processed_dir / "processing_progress.json", "w") as f:
            json.dump(self.progress, f, indent=2)

    def _validate_audio(self, audio_path):
        """Validate audio file exists and is not empty."""
        if not audio_path.exists():
            logger.warning(f"File not found: {audio_path}")
            return False
        if audio_path.stat().st_size == 0:
            logger.warning(f"Empty file: {audio_path}")
            return False
        return True

    def _normalize_features(self, features):
        """Normalize features using StandardScaler.
        
        Args:
            features (dict): Dictionary of features to normalize
            
        Returns:
            dict: Dictionary of normalized features
        """
        # Normalize mel spectrogram
        mel_shape = features["mel_spectrogram"].shape
        mel_flat = features["mel_spectrogram"].reshape(-1, mel_shape[-1])
        mel_norm = self.mel_scaler.fit_transform(mel_flat)
        features["mel_spectrogram"] = mel_norm.reshape(mel_shape)
        
        # Normalize MFCCs
        mfcc_shape = features["mfcc"].shape
        mfcc_flat = features["mfcc"].reshape(-1, mfcc_shape[-1])
        mfcc_norm = self.mfcc_scaler.fit_transform(mfcc_flat)
        features["mfcc"] = mfcc_norm.reshape(mfcc_shape)
        
        # Normalize audio characteristics
        audio_features = np.array([
            features["tempo"],
            features["loudness"],
            features["brightness"],
            features["noisiness"]
        ]).reshape(1, -1)
        audio_norm = self.audio_scaler.fit_transform(audio_features)
        features["tempo"] = audio_norm[0, 0]
        features["loudness"] = audio_norm[0, 1]
        features["brightness"] = audio_norm[0, 2]
        features["noisiness"] = audio_norm[0, 3]
        
        return features

    def _pad_or_truncate(self, features):
        """Pad or truncate features to a fixed length.
        
        Args:
            features (dict): Dictionary of features to process
            
        Returns:
            dict: Dictionary of processed features
        """
        # Process mel spectrogram
        mel_spec = features["mel_spectrogram"]
        if mel_spec.shape[-1] > self.max_length:
            mel_spec = mel_spec[:, :, :self.max_length]
        else:
            pad_width = ((0, 0), (0, 0), (0, self.max_length - mel_spec.shape[-1]))
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
        features["mel_spectrogram"] = mel_spec
        
        # Process MFCCs
        mfcc = features["mfcc"]
        if mfcc.shape[-1] > self.max_length:
            mfcc = mfcc[:, :, :self.max_length]
        else:
            pad_width = ((0, 0), (0, 0), (0, self.max_length - mfcc.shape[-1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        features["mfcc"] = mfcc
        
        return features

    def _extract_features(self, waveform, sample_rate):
        """Extract audio features using torchaudio transforms and librosa.
        
        Args:
            waveform (torch.Tensor): Audio waveform
            sample_rate (int): Sample rate of the audio
            
        Returns:
            dict: Dictionary containing extracted features
        """
        try:
            logger.info(f"Input waveform shape: {waveform.shape}, dtype: {waveform.dtype}")
            logger.info(f"Input sample rate: {sample_rate}")
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                logger.info(f"Resampling from {sample_rate} to {self.sample_rate}")
                resampler = T.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                logger.info("Converting stereo to mono")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to numpy for librosa processing
            audio_np = waveform.numpy().squeeze()
            logger.info(f"Numpy array shape: {audio_np.shape}, dtype: {audio_np.dtype}")
            
            # Extract mel spectrogram
            mel_spec = self.mel_transform(waveform)
            logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
            
            # Extract MFCCs
            mfcc = self.mfcc_transform(waveform)
            logger.info(f"MFCC shape: {mfcc.shape}")
            
            # Calculate additional characteristics using librosa
            # BPM estimation
            logger.info("Calculating tempo...")
            onset_env = librosa.onset.onset_strength(y=audio_np, sr=self.sample_rate)
            tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=self.sample_rate)[0]
            logger.info(f"Estimated tempo: {tempo}")
            
            # Loudness (RMS energy)
            loudness = torch.sqrt(torch.mean(waveform ** 2))
            logger.info(f"Calculated loudness: {loudness.item()}")
            
            # Brightness (spectral centroid)
            spec = torch.stft(waveform[0], self.n_fft, self.hop_length, return_complex=True)
            freqs = torch.linspace(0, self.sample_rate/2, self.n_fft//2 + 1)
            magnitudes = torch.abs(spec)
            brightness = torch.sum(freqs[:, None] * magnitudes) / torch.sum(magnitudes)
            logger.info(f"Calculated brightness: {brightness.item()}")
            
            # Zero crossing rate (measure of noisiness)
            zero_crossings = torch.sum(torch.diff(torch.sign(waveform[0])) != 0)
            noisiness = zero_crossings / waveform.shape[1]
            logger.info(f"Calculated noisiness: {noisiness.item()}")
            
            features = {
                "mel_spectrogram": mel_spec.numpy(),
                "mfcc": mfcc.numpy(),
                "tempo": tempo,
                "loudness": loudness.item(),
                "brightness": brightness.item(),
                "noisiness": noisiness.item()
            }
            
            # Normalize and pad/truncate features
            features = self._normalize_features(features)
            features = self._pad_or_truncate(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            raise

    def process_audio(self, audio_path, surface_type):
        """Process a single audio file and extract features.
        
        Args:
            audio_path (Path): Path to the audio file
            surface_type (str): Type of surface the rain is falling on
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Skip if already processed
            if str(audio_path) in self.progress["processed"]:
                logger.info(f"Skipping already processed file: {audio_path}")
                return True
                
            # Skip if previously failed
            if str(audio_path) in self.progress["failed"]:
                logger.info(f"Skipping previously failed file: {audio_path}")
                return False
            
            # Validate audio file
            if not self._validate_audio(audio_path):
                self.progress["failed"].append(str(audio_path))
                return False
            
            logger.info(f"Processing file: {audio_path}")
            logger.info(f"Surface type: {surface_type}")
            
            # Load audio file
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                logger.info(f"Successfully loaded audio: shape={waveform.shape}, sr={sample_rate}")
            except Exception as e:
                logger.error(f"Error loading audio file: {str(e)}")
                logger.error("Traceback:")
                logger.error(traceback.format_exc())
                self.progress["failed"].append(str(audio_path))
                self._save_progress()
                return False
            
            # Extract features
            try:
                features = self._extract_features(waveform, sample_rate)
                logger.info("Successfully extracted features")
            except Exception as e:
                logger.error(f"Error extracting features: {str(e)}")
                logger.error("Traceback:")
                logger.error(traceback.format_exc())
                self.progress["failed"].append(str(audio_path))
                self._save_progress()
                return False
            
            # Save features
            try:
                output_file = self.processed_dir / f"{surface_type}_{audio_path.stem}.npz"
                np.savez(
                    output_file,
                    surface_type=surface_type,
                    **features
                )
                logger.info(f"Saved features to {output_file}")
            except Exception as e:
                logger.error(f"Error saving features: {str(e)}")
                logger.error("Traceback:")
                logger.error(traceback.format_exc())
                self.progress["failed"].append(str(audio_path))
                self._save_progress()
                return False
            
            # Update progress
            self.progress["processed"].append(str(audio_path))
            self._save_progress()
            
            logger.info(f"Successfully processed {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            self.progress["failed"].append(str(audio_path))
            self._save_progress()
            return False

    def create_dataset(self):
        """Process all audio files in the raw directory and create the dataset."""
        # Get all surface type directories
        surface_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir() and d.name != "metadata"]
        
        # Process each surface type
        for surface_dir in surface_dirs:
            surface_type = surface_dir.name
            logger.info(f"Processing {surface_type} sounds...")
            
            # Get all audio files for this surface type
            audio_files = list(surface_dir.glob("*.wav")) + list(surface_dir.glob("*.mp3"))
            logger.info(f"Found {len(audio_files)} files in {surface_type}")
            
            # Process each audio file with progress bar
            for audio_file in tqdm(audio_files, desc=f"Processing {surface_type}"):
                self.process_audio(audio_file, surface_type)
        
        logger.info("Dataset creation completed!")
        logger.info(f"Processed files: {len(self.progress['processed'])}")
        logger.info(f"Failed files: {len(self.progress['failed'])}")

def extract_features(audio, sr):
    """
    Extract audio features using torchaudio and librosa.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
    
    Returns:
        dict: Dictionary containing extracted features
    """
    try:
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure audio is float and in [-1, 1] range
        if audio.abs().max() > 1:
            audio = audio / audio.abs().max()
            
        # Resample if needed
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            audio = resampler(audio)
            sr = 16000
            
        # Extract mel spectrogram
        mel_spec_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        mel_spec = mel_spec_transform(audio)
        
        # Extract MFCC
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={'n_fft': 1024, 'hop_length': 512, 'n_mels': 64}
        )
        mfcc = mfcc_transform(audio)
        
        # Extract additional characteristics using librosa
        audio_np = audio.numpy()
        
        # Tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=audio_np, sr=sr)
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Loudness (RMS energy)
        loudness = float(torch.sqrt(torch.mean(audio ** 2)).item())
        
        # Spectral centroid (brightness)
        spec_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sr).mean()
        
        # Spectral flatness (noisiness)
        spec_flatness = librosa.feature.spectral_flatness(y=audio_np).mean()
        
        # Return all features
        return {
            'mel_spectrogram': mel_spec.numpy(),
            'mfcc': mfcc.numpy(),
            'tempo': tempo,
            'loudness': loudness,
            'brightness': spec_centroid,
            'noisiness': spec_flatness
        }
        
    except Exception as e:
        logging.error(f"Error extracting features: {str(e)}")
        raise

def main():
    """Main function to process all audio files."""
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Get all surface types
    surface_types = [d for d in os.listdir('data/raw/rain_sounds') 
                    if os.path.isdir(os.path.join('data/raw/rain_sounds', d))]
    
    # Create label map
    label_map = {label: idx for idx, label in enumerate(sorted(surface_types))}
    
    # Save label map
    with open('data/processed/label_map.json', 'w') as f:
        json.dump(label_map, f)
    
    logging.info(f"Found {len(surface_types)} surface types: {surface_types}")
    
    # Process each surface type
    total_files = 0
    failed_files = 0
    
    for surface_type in surface_types:
        surface_dir = os.path.join('data/raw/rain_sounds', surface_type)
        files = [f for f in os.listdir(surface_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        
        logging.info(f"Processing {surface_type} sounds: {len(files)} files found")
        
        for file in files:
            try:
                file_path = os.path.join(surface_dir, file)
                logging.info(f"Processing {file_path}")
                
                # Load and process audio
                audio, sr = librosa.load(file_path, sr=16000, mono=True)
                
                # Extract features
                features = extract_features(audio, sr)
                
                # Add metadata
                features['label'] = label_map[surface_type]
                features['surface_type'] = surface_type
                features['filename'] = file
                
                # Save features
                output_file = os.path.join('data/processed', f"{surface_type}_{os.path.splitext(file)[0]}.npz")
                np.savez(output_file, **features)
                
                total_files += 1
                logging.info(f"Successfully processed {file_path}")
                
            except Exception as e:
                failed_files += 1
                logging.error(f"Failed to process {file_path}: {str(e)}")
    
    logging.info(f"Dataset creation completed. Processed {total_files} files, {failed_files} failed.")

if __name__ == "__main__":
    main() 