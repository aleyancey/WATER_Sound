import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List
import shutil
from tqdm import tqdm

def analyze_rain_intensity(audio_path: str) -> Tuple[float, float, float]:
    """
    Analyze the intensity of a rain sound based on audio features.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Tuple of (rms_energy, spectral_centroid, zero_crossing_rate)
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate RMS energy (volume)
    rms = librosa.feature.rms(y=y)[0].mean()
    
    # Calculate spectral centroid (brightness/intensity)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    
    # Calculate zero crossing rate (noisiness)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    
    return rms, spectral_centroid, zero_crossing_rate

def determine_intensity(features: Tuple[float, float, float]) -> str:
    """
    Determine rain intensity based on audio features.
    
    Args:
        features: Tuple of (rms_energy, spectral_centroid, zero_crossing_rate)
        
    Returns:
        String indicating intensity ('light', 'moderate', or 'heavy')
    """
    rms, spectral_centroid, zcr = features
    
    # Normalize features
    rms_norm = np.clip(rms / 0.1, 0, 1)  # Typical RMS values are 0-0.1
    sc_norm = np.clip(spectral_centroid / 5000, 0, 1)  # Typical centroid values are 0-5000
    zcr_norm = np.clip(zcr / 0.2, 0, 1)  # Typical ZCR values are 0-0.2
    
    # Calculate overall intensity score
    intensity_score = (rms_norm * 0.5 + sc_norm * 0.3 + zcr_norm * 0.2)
    
    # Classify based on score
    if intensity_score < 0.4:
        return "light"
    elif intensity_score < 0.7:
        return "moderate"
    else:
        return "heavy"

def process_rain_sounds(input_dir: str, output_dir: str):
    """
    Process rain sound files and organize them by intensity.
    
    Args:
        input_dir: Directory containing input rain sound files
        output_dir: Directory to save organized files
    """
    # Create output directories
    for intensity in ["light", "moderate", "heavy"]:
        os.makedirs(os.path.join(output_dir, intensity), exist_ok=True)
    
    # Process each audio file
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3'))]
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Processing {len(audio_files)} audio files...")
    
    intensity_counts = {"light": 0, "moderate": 0, "heavy": 0}
    
    for filename in tqdm(audio_files):
        input_path = os.path.join(input_dir, filename)
        
        try:
            # Analyze the audio file
            features = analyze_rain_intensity(input_path)
            intensity = determine_intensity(features)
            
            # Copy file to appropriate directory
            output_path = os.path.join(output_dir, intensity, filename)
            shutil.copy2(input_path, output_path)
            
            intensity_counts[intensity] += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Print summary
    print("\nProcessing complete!")
    print("\nFile distribution:")
    for intensity, count in intensity_counts.items():
        print(f"{intensity.capitalize()} rain: {count} files")
    
    print(f"\nFiles organized in: {output_dir}")

if __name__ == "__main__":
    # Define directories
    input_dir = "data/raw/rain_sounds"
    output_dir = "data/processed/rain_sounds"
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Check if input directory is empty
    if not os.listdir(input_dir):
        print(f"\nPlease add rain sound files to {input_dir}")
        print("You can download free rain sounds from sources like:")
        print("1. https://freesound.org/search/?q=rain")
        print("2. https://soundbible.com/search.php?q=rain")
        print("3. https://www.zapsplat.com/sound-effect-category/rain/")
    else:
        process_rain_sounds(input_dir, output_dir) 