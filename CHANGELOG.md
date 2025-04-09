# Changelog

## [Unreleased]

### Added
- Created `prepare_dataset.py` script for dataset preparation and processing
  - Implemented `RainSoundDataset` class for handling audio processing and feature extraction
  - Added support for loading and processing metadata
  - Integrated Wav2Vec2 feature extractor for audio feature extraction
  - Implemented train/validation split functionality
  - Added error handling and logging for audio processing

### Changed
- Updated `download_freesound.py` to include metadata collection
  - Added support for multiple surface types
  - Implemented metadata storage in JSON format
  - Added progress tracking for downloads

### Fixed
- Fixed authentication issues in Freesound API integration
- Improved error handling in audio processing pipeline

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