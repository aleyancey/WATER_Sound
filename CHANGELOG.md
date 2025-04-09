# Changelog

## [Unreleased]

### Added
- Created `prepare_dataset.py` script for dataset preparation and processing
  - Implemented `RainSoundDataset` class for handling audio processing and feature extraction
  - Added support for loading and processing metadata
  - Integrated Wav2Vec2 feature extractor for audio feature extraction
  - Implemented train/validation split functionality
  - Added error handling and logging for audio processing
- Proper logging setup in prepare_dataset.py using Python's logging module
- New processed_files list to track successfully processed audio files
- Better error handling and logging throughout the audio processing pipeline
- Added robust audio file validation using `soundfile` library
- Added support for additional audio formats (mp3, wav, ogg, flac, m4a, aac)
- Added detailed logging with timestamps and log levels
- Added temporary directory for audio conversion process
- Added audio normalization step in processing pipeline
- Progress tracking and recovery system in `prepare_dataset.py`
  - Saves processing progress to JSON file
  - Tracks failed files with error messages
  - Allows resuming interrupted processing
- Improved validation and error handling in audio processing pipeline
  - Detailed error messages and logging
  - Robust audio file validation
  - Better FFmpeg integration for audio conversion
- Support for multiple audio formats (.wav, .mp3, .flac, .ogg, .m4a)
- Progress bar for dataset processing using tqdm
- Temporary directory management for audio conversion
- Configurable model name for feature extractor
- Created fine-tuning script `fine_tune_wav2vec2.py`:
  - Support for surface type classification using Wav2Vec2
  - Custom data collator for audio classification
  - Configurable training parameters
  - Progress tracking and model checkpointing
  - Evaluation metrics and logging
  - TensorBoard integration for monitoring training

### Changed
- Updated `download_freesound.py` to include metadata collection
  - Added support for multiple surface types
  - Implemented metadata storage in JSON format
  - Added progress tracking for downloads
- Improved audio file processing in prepare_dataset.py:
  - Better handling of WAV conversion
  - More robust feature extraction
  - Cleaner dataset creation process
- Updated dataset structure to include more metadata
- Replaced print statements with proper logging
- Made train/validation split ratio configurable
- Improved FFmpeg conversion process with better error handling and subprocess management
- Enhanced audio processing pipeline with more robust validation and error checks
- Updated logging to use proper logger instead of print statements
- Improved cleanup of temporary files
- Made audio format handling more flexible and extensible
- Enhanced error messages to be more descriptive and helpful
- Enhanced RainSoundDataset class with better organization and documentation
- Improved audio processing pipeline with better error handling
- More robust metadata loading with graceful fallback
- Optimized audio conversion process
- Better memory management for large datasets
- More informative logging messages
- Enhanced feature extraction in `prepare_dataset.py`:
  - Improved audio format handling and validation
  - Added detailed logging of audio properties before and after feature extraction
  - Fixed audio dimensionality issues by ensuring 1D input
  - Added max_length padding for consistent feature dimensions
  - Improved error messages with detailed audio state information
- Improved dataset preparation in `prepare_dataset.py`:
  - Fixed feature extraction issues
  - Added proper audio normalization
  - Improved error handling and logging
  - Added support for resuming interrupted processing
  - Fixed progress tracking serialization

### Fixed
- Fixed authentication issues in Freesound API integration
- Improved error handling in audio processing pipeline
- Audio processing pipeline now properly handles stereo files
- Feature extraction now correctly processes all valid audio files
- Dataset creation now includes proper error handling and validation
- Fixed potential issues with audio file conversion
- Fixed memory leaks by properly cleaning up temporary files
- Fixed error handling in feature extraction process
- Fixed audio validation to catch empty or corrupt files early
- Fixed metadata loading to be more robust
- Issue with metadata directory handling when not present
- Memory leaks in audio processing pipeline
- Progress tracking persistence
- Temporary file cleanup
- Audio format validation
- Fixed AttributeError in feature extraction by properly handling audio dimensions
- Improved attention mask handling in feature extraction
- Fixed feature extraction pipeline in dataset preparation
- Resolved JSON serialization issues with numpy arrays
- Improved error handling in audio processing

- Added additional sound samples to data/processed/sound_samples:
  - Rain and ambient sounds: gentle rain on leaves with suburban ambience
  - Creek sounds: gentle creek in rain forest with cicadas
  - Bird sounds (MP3 format)
  - General rain sounds 

- Added audio effects processing capabilities:
  - Delay effect with customizable delay time and decay
  - Reverb effect with adjustable room size and damping
  - Test script to demonstrate effects on water sound samples
  - Required dependencies in requirements.txt

- Added wav2vec2 fine-tuning setup:
  - Created fine_tune_wav2vec2.py for model training
  - Implemented custom WaterSoundDataset class
  - Added training configuration for CPU
  - Updated requirements.txt with necessary dependencies
  - Set up logging and model saving functionality 

## [0.2.0] - 2025-04-09

### Changed
- Switched from YAMNET to PyTorch/torchaudio for feature extraction
- Updated feature extraction to include:
  - Mel spectrogram
  - MFCC (Mel-frequency cepstral coefficients)
  - Tempo (BPM)
  - Loudness (RMS energy)
  - Brightness (spectral centroid)
  - Noisiness (spectral flatness)
- Fixed deprecated librosa.beat.tempo call
- Improved error handling and logging
- Added metadata to processed files (label, surface type, filename)

### Added
- Label mapping file generation (data/processed/label_map.json)
- Support for multiple audio formats (wav, mp3, flac)
- Automatic audio resampling to 16kHz
- Audio normalization to [-1, 1] range

## [0.1.0] - 2025-04-08

### Added
- Initial project setup
- Basic audio processing pipeline
- Support for different surface types
- Basic error handling 