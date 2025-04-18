import os
import librosa
from pathlib import Path

def check_audio_durations():
    # Base directory containing audio files
    base_dir = Path("data/raw/rain_sounds")
    
    # Dictionary to store durations
    durations = {}
    
    # Check files in the root directory
    for file in base_dir.glob("*.wav"):
        try:
            duration = librosa.get_duration(path=str(file))
            durations[file.name] = duration
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Check files in subdirectories
    for subdir in base_dir.iterdir():
        if subdir.is_dir() and subdir.name != "metadata":
            for file in subdir.glob("*.wav"):
                try:
                    duration = librosa.get_duration(path=str(file))
                    durations[f"{subdir.name}/{file.name}"] = duration
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    # Print results
    print("\nAudio File Durations:")
    print("-" * 50)
    for file, duration in durations.items():
        print(f"{file}: {duration:.2f} seconds")
    
    # Print summary
    print("\nSummary:")
    print(f"Total files checked: {len(durations)}")
    print(f"Average duration: {sum(durations.values()) / len(durations):.2f} seconds")
    print(f"Shortest duration: {min(durations.values()):.2f} seconds")
    print(f"Longest duration: {max(durations.values()):.2f} seconds")

if __name__ == "__main__":
    check_audio_durations() 