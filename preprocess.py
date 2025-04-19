import os
import subprocess
from pathlib import Path

# Set your source and destination directories
RAW_ROOT = Path("data/raw/rain_sounds")
PROCESSED_ROOT = Path("data/processed/rain_sounds")

# Supported audio extensions
AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aac"]

# Target normalization peak (in dBFS)
TARGET_DBFS = -1.0  # -1 dBFS is a safe standard

def normalize_and_convert(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Use ffmpeg to convert and normalize
    # -af "dynaudnorm" is a good simple normalization filter, or use "loudnorm" for EBU R128
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite existing
        "-i", str(src_path),
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        "-af", f"loudnorm=I=-16:TP={TARGET_DBFS}:LRA=11",
        str(dst_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Processed: {src_path} -> {dst_path}")
    else:
        print(f"Error processing {src_path}: {result.stderr}")

def main():
    for root, _, files in os.walk(RAW_ROOT):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in AUDIO_EXTS):
                src_path = Path(root) / fname
                # Mirror the folder structure under processed/
                rel_path = src_path.relative_to(RAW_ROOT)
                dst_path = PROCESSED_ROOT / rel_path.with_suffix(".wav")
                normalize_and_convert(src_path, dst_path)

if __name__ == "__main__":
    import numpy as np
    import soundfile as sf
    
    # Intensity thresholds (RMS, adjust as needed after testing)
    LIGHT_THRESHOLD = 0.03
    MODERATE_THRESHOLD = 0.08
    # Anything above MODERATE_THRESHOLD is 'heavy'

    intensity_counts = {"light": 0, "moderate": 0, "heavy": 0}

    def compute_rms(wav_path):
        data, _ = sf.read(wav_path)
        # If stereo, average channels
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        rms = np.sqrt(np.mean(np.square(data)))
        return float(rms)

    for root, _, files in os.walk(RAW_ROOT):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in AUDIO_EXTS):
                src_path = Path(root) / fname
                rel_path = src_path.relative_to(RAW_ROOT)
                # Temporary normalized wav output (in memory location)
                tmp_wav = Path("/tmp") / (fname + ".norm.wav")
                normalize_and_convert(src_path, tmp_wav)
                # Analyze intensity
                rms = compute_rms(tmp_wav)
                if rms < LIGHT_THRESHOLD:
                    intensity = "light"
                elif rms < MODERATE_THRESHOLD:
                    intensity = "moderate"
                else:
                    intensity = "heavy"
                # Mirror subfolder structure inside intensity folder
                dst_path = PROCESSED_ROOT / intensity / rel_path.with_suffix(".wav")
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_wav, dst_path)
                intensity_counts[intensity] += 1
                print(f"{src_path} -> {dst_path} (RMS={rms:.3f}, {intensity})")
    print("\nSummary:")
    for k, v in intensity_counts.items():
        print(f"{k.capitalize()}: {v} files")

    print(f"An error occurred: {e}")