# WATER_Sound - Interactive Rain Soundscape Generator

An interactive application for creating and manipulating rain soundscapes. Users can mix different types of rain sounds, apply effects, and generate evolving soundscapes.

## Features

### Sound Selection and Mixing
- Select up to 4 different rain types:
  - Wood
  - Water
  - Metal
  - Concrete
  - Grass
  - Leaves
  - Heavy rain
  - Glass
  - Medium rain
- Adjust individual volume levels
- Crossfade between sounds
- Real-time mixing

### Effects Processing
- Delay
  - Time control
  - Feedback
  - Mix level
- Reverb
  - Room size
  - Damping
  - Mix level
- Low-pass/High-pass filters
- Dynamic range compression

### Sound Characteristics
- Tempo (BPM) adjustment
- Loudness control
- Brightness modification
- Noisiness adjustment

### UI Components

#### Main Screen
```
+------------------------------------------+
|  Title Bar                               |
+------------------------------------------+
|  [Sound Selection Panel]  [Waveform]     |
|  [Rain Type 1] [Vol] [Mute] [Solo]      |
|  [Rain Type 2] [Vol] [Mute] [Solo]      |
|  [Rain Type 3] [Vol] [Mute] [Solo]      |
|  [Rain Type 4] [Vol] [Mute] [Solo]      |
|                                          |
|  [Waveform Display]                      |
|  [Time Scale]                            |
+------------------------------------------+
|  [Effects Panel]                         |
|  [Delay] [Reverb] [Filter] [Compressor]  |
|  [Effect Parameters]                     |
+------------------------------------------+
|  [Characteristics Panel]                 |
|  [Tempo] [Loudness] [Brightness] [Noise] |
+------------------------------------------+
|  [Transport Controls]                    |
|  [Play] [Stop] [Loop] [Generate]         |
+------------------------------------------+
```

#### Sound Selection Panel
- Dropdown menu for each sound slot
- Volume fader
- Mute/Solo buttons
- Crossfade controls between adjacent sounds

#### Waveform Display
- Real-time waveform visualization
- Zoom controls
- Time markers
- Playhead position

#### Effects Panel
- Effect chain visualization
- Parameter controls for each effect
- Preset management
- Bypass switches

#### Characteristics Panel
- Sliders for each characteristic
- Visual feedback of current values
- Preset management
- Randomize button

#### Transport Controls
- Play/Stop button
- Loop toggle
- Generate new variation button
- Save/Load presets

## Technical Implementation

### Backend
- Python with PyTorch for audio processing
- librosa for feature extraction
- sounddevice for real-time audio
- numpy for signal processing

### Frontend
- PyQt6 for the user interface
- matplotlib for visualizations
- Custom audio processing pipeline

### Audio Processing Pipeline
1. Input selection and mixing
2. Feature extraction and analysis
3. Effects processing
4. Real-time synthesis
5. Output routing

## Development Roadmap

### Phase 1: Core Functionality
- [x] Dataset preparation
- [x] Feature extraction
- [ ] Basic UI implementation
- [ ] Sound mixing engine
- [ ] Effect processing

### Phase 2: Advanced Features
- [ ] Real-time visualization
- [ ] Preset management
- [ ] Sound generation
- [ ] Export functionality

### Phase 3: Polish
- [ ] UI/UX improvements
- [ ] Performance optimization
- [ ] Documentation
- [ ] Testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WATER_Sound.git
cd WATER_Sound
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python src/main.py
```

2. Select rain types and adjust parameters
3. Mix and process sounds
4. Generate and save soundscapes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 