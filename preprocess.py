import subprocess
import os

# Define input and output file paths
# Using relative paths from the project root
m4a_file = "Creek_test.m4a"  # Input M4A file in the project root
wav_file = "data/processed/Creek_test.wav"  # Output WAV file in the processed data directory

# Create the processed directory if it doesn't exist
os.makedirs(os.path.dirname(wav_file), exist_ok=True)

try:
    # Convert M4A to WAV using ffmpeg
    cmd = ['ffmpeg', '-i', m4a_file, '-acodec', 'pcm_s16le', '-ar', '44100', wav_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully converted '{m4a_file}' to '{wav_file}'")
    else:
        print(f"Error during conversion: {result.stderr}")
except FileNotFoundError:
    print(f"Error: The file '{m4a_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")