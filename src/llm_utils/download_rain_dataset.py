import os
from datasets import load_dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm
import librosa

def download_rain_dataset():
    """
    Download and prepare rain sound datasets from Hugging Face.
    This function downloads multiple rain sound datasets and organizes them by intensity.
    """
    # Create output directories
    output_dir = "data/processed/rain_sounds"
    os.makedirs(os.path.join(output_dir, "light"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "moderate"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "heavy"), exist_ok=True)
    
    print("Downloading FSD50K dataset...")
    
    try:
        # Load FSD50K dataset
        dataset = load_dataset("DCASE/FSD50K", split="train")
        
        # Keywords for different rain intensities
        intensity_keywords = {
            "light": ["drizzle", "light rain", "gentle rain"],
            "moderate": ["rain", "rainfall", "steady rain"],
            "heavy": ["heavy rain", "downpour", "storm", "thunder"]
        }
        
        # Process each intensity level
        for intensity, keywords in intensity_keywords.items():
            print(f"\nProcessing {intensity} rain sounds...")
            
            # Filter for rain sounds of this intensity
            for keyword in keywords:
                filtered_dataset = dataset.filter(
                    lambda x: any(keyword in label.lower() for label in x["labels"])
                )
                
                # Process each audio sample
                for idx, sample in enumerate(tqdm(filtered_dataset, desc=f"Processing {keyword}")):
                    try:
                        # Get audio data
                        audio_path = sample["audio"]["path"]
                        audio_data, sample_rate = librosa.load(audio_path, sr=None)
                        
                        # Create output filename
                        output_filename = f"{intensity}_{keyword.replace(' ', '_')}_{idx}.wav"
                        output_path = os.path.join(output_dir, intensity, output_filename)
                        
                        # Save audio file
                        sf.write(output_path, audio_data, sample_rate)
                        
                    except Exception as e:
                        print(f"Error processing audio file {audio_path}: {str(e)}")
                        continue
        
    except Exception as e:
        print(f"Error accessing FSD50K dataset: {str(e)}")
    
    # Count files in each directory
    for intensity in ["light", "moderate", "heavy"]:
        dir_path = os.path.join(output_dir, intensity)
        num_files = len([f for f in os.listdir(dir_path) if f.endswith('.wav')])
        print(f"\nFound {num_files} {intensity} rain sound files")
    
    print("\nDataset processing complete!")
    print(f"Files saved in: {output_dir}")
    print("\nDirectory structure:")
    print(f"- {os.path.join(output_dir, 'light')}: Light rain/drizzle sounds")
    print(f"- {os.path.join(output_dir, 'moderate')}: Moderate rain sounds")
    print(f"- {os.path.join(output_dir, 'heavy')}: Heavy rain/thunder sounds")

if __name__ == "__main__":
    download_rain_dataset() 