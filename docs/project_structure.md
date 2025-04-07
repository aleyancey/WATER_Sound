# Project Structure

```
WATER_Sound/
├── src/                    # Source code directory
│   ├── audio_processing/   # Audio processing modules
│   ├── llm_utils/         # LLM-related utilities and fine-tuning code
│   ├── config/            # Configuration files
│   └── utils/             # General utility functions
├── data/                  # Data directory
│   ├── raw/              # Raw audio files
│   └── processed/        # Processed audio files
├── tests/                # Test directory
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── docs/                # Documentation
└── readme.md           # Project overview
```

## Directory Descriptions

- `src/audio_processing/`: Contains modules for audio file handling, effects application (pulsate, echo), and audio processing utilities
- `src/llm_utils/`: Contains code for LLM fine-tuning, model configuration, and PEFT implementation
- `src/config/`: Configuration files for the project, including model parameters and processing settings
- `src/utils/`: General utility functions used across the project
- `data/raw/`: Original water sound samples
- `data/processed/`: Modified audio files after processing
- `tests/`: Test files organized by unit and integration tests
- `docs/`: Project documentation, including setup guides and API references 