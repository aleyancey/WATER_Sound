import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
                           QGroupBox, QFrame)
from PyQt6.QtCore import Qt, QTimer
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SoundMixer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WATER_Sound - Rain Soundscape Generator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize audio parameters
        self.sample_rate = 44100
        self.buffer_size = 1024
        self.current_sounds = [None] * 4
        self.volumes = [1.0] * 4
        self.muted = [False] * 4
        self.soloed = [False] * 4
        self.current_positions = [0] * 4
        self.waveform_data = np.zeros(self.buffer_size)  # Buffer for waveform display
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create sound selection panel
        self.create_sound_selection_panel(layout)
        
        # Create waveform display
        self.create_waveform_display(layout)
        
        # Create effects panel
        self.create_effects_panel(layout)
        
        # Create characteristics panel
        self.create_characteristics_panel(layout)
        
        # Create transport controls
        self.create_transport_controls(layout)
        
        # Initialize audio stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=self.audio_callback
        )
        
        # Setup waveform update timer
        self.waveform_timer = QTimer()
        self.waveform_timer.timeout.connect(self.update_waveform)
        self.waveform_timer.start(50)  # Update every 50ms
        
    def create_sound_selection_panel(self, parent_layout):
        group = QGroupBox("Sound Selection")
        layout = QVBoxLayout()
        
        # Create 4 sound slots
        for i in range(4):
            slot_layout = QHBoxLayout()
            
            # Sound type selection
            combo = QComboBox()
            combo.addItems(["None", "Wood", "Water", "Metal", "Concrete", 
                          "Grass", "Leaves", "Heavy rain", "Glass", "Medium rain"])
            combo.currentIndexChanged.connect(lambda idx, slot=i: self.sound_selected(idx, slot))
            
            # Volume slider
            volume = QSlider(Qt.Orientation.Horizontal)
            volume.setRange(0, 100)
            volume.setValue(100)
            volume.valueChanged.connect(lambda val, slot=i: self.volume_changed(val, slot))
            
            # Mute button
            mute = QPushButton("Mute")
            mute.setCheckable(True)
            mute.toggled.connect(lambda checked, slot=i: self.mute_toggled(checked, slot))
            
            # Solo button
            solo = QPushButton("Solo")
            solo.setCheckable(True)
            solo.toggled.connect(lambda checked, slot=i: self.solo_toggled(checked, slot))
            
            slot_layout.addWidget(combo)
            slot_layout.addWidget(volume)
            slot_layout.addWidget(mute)
            slot_layout.addWidget(solo)
            layout.addLayout(slot_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_waveform_display(self, parent_layout):
        group = QGroupBox("Waveform Display")
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 2))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.buffer_size)
        self.ax.grid(True)
        self.waveform_line, = self.ax.plot(np.zeros(self.buffer_size))
        
        layout.addWidget(self.canvas)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_effects_panel(self, parent_layout):
        group = QGroupBox("Effects")
        layout = QHBoxLayout()
        
        # Delay controls
        delay_group = QGroupBox("Delay")
        delay_layout = QVBoxLayout()
        self.delay_time = QSlider(Qt.Orientation.Horizontal)
        self.delay_feedback = QSlider(Qt.Orientation.Horizontal)
        self.delay_mix = QSlider(Qt.Orientation.Horizontal)
        delay_layout.addWidget(QLabel("Time"))
        delay_layout.addWidget(self.delay_time)
        delay_layout.addWidget(QLabel("Feedback"))
        delay_layout.addWidget(self.delay_feedback)
        delay_layout.addWidget(QLabel("Mix"))
        delay_layout.addWidget(self.delay_mix)
        delay_group.setLayout(delay_layout)
        
        # Reverb controls
        reverb_group = QGroupBox("Reverb")
        reverb_layout = QVBoxLayout()
        self.reverb_room = QSlider(Qt.Orientation.Horizontal)
        self.reverb_damping = QSlider(Qt.Orientation.Horizontal)
        self.reverb_mix = QSlider(Qt.Orientation.Horizontal)
        reverb_layout.addWidget(QLabel("Room Size"))
        reverb_layout.addWidget(self.reverb_room)
        reverb_layout.addWidget(QLabel("Damping"))
        reverb_layout.addWidget(self.reverb_damping)
        reverb_layout.addWidget(QLabel("Mix"))
        reverb_layout.addWidget(self.reverb_mix)
        reverb_group.setLayout(reverb_layout)
        
        layout.addWidget(delay_group)
        layout.addWidget(reverb_group)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_characteristics_panel(self, parent_layout):
        group = QGroupBox("Sound Characteristics")
        layout = QHBoxLayout()
        
        # Tempo control
        tempo_group = QGroupBox("Tempo")
        tempo_layout = QVBoxLayout()
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setRange(40, 200)
        self.tempo_slider.setValue(120)
        tempo_layout.addWidget(self.tempo_slider)
        tempo_group.setLayout(tempo_layout)
        
        # Loudness control
        loudness_group = QGroupBox("Loudness")
        loudness_layout = QVBoxLayout()
        self.loudness_slider = QSlider(Qt.Orientation.Horizontal)
        loudness_layout.addWidget(self.loudness_slider)
        loudness_group.setLayout(loudness_layout)
        
        # Brightness control
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout()
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_group.setLayout(brightness_layout)
        
        # Noisiness control
        noise_group = QGroupBox("Noisiness")
        noise_layout = QVBoxLayout()
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        noise_layout.addWidget(self.noise_slider)
        noise_group.setLayout(noise_layout)
        
        layout.addWidget(tempo_group)
        layout.addWidget(loudness_group)
        layout.addWidget(brightness_group)
        layout.addWidget(noise_group)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_transport_controls(self, parent_layout):
        layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self.play_toggled)
        
        self.loop_button = QPushButton("Loop")
        self.loop_button.setCheckable(True)
        
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_soundscape)
        
        layout.addWidget(self.play_button)
        layout.addWidget(self.loop_button)
        layout.addWidget(self.generate_button)
        
        parent_layout.addLayout(layout)
        
    def sound_selected(self, index, slot):
        if index == 0:
            self.current_sounds[slot] = None
        else:
            # Load the selected sound
            sound_type = self.sender().currentText().lower().replace(" ", "_")
            try:
                sound_path = f"data/raw/rain_sounds/{sound_type}/"
                if os.path.exists(sound_path):
                    files = [f for f in os.listdir(sound_path) if f.endswith(('.wav', '.mp3'))]
                    if files:
                        self.current_sounds[slot] = librosa.load(
                            os.path.join(sound_path, files[0]),
                            sr=self.sample_rate
                        )[0]
            except Exception as e:
                print(f"Error loading sound: {e}")
                
    def volume_changed(self, value, slot):
        self.volumes[slot] = value / 100.0
        
    def mute_toggled(self, checked, slot):
        self.muted[slot] = checked
        
    def solo_toggled(self, checked, slot):
        self.soloed[slot] = checked
        
    def play_toggled(self, checked):
        if checked:
            self.stream.start()
        else:
            self.stream.stop()
            
    def generate_soundscape(self):
        # TODO: Implement sound generation
        pass
        
    def update_waveform(self):
        if hasattr(self, 'waveform_data'):
            self.waveform_line.set_ydata(self.waveform_data)
            self.canvas.draw()
            
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
            
        # Mix all active sounds
        mixed = np.zeros((frames, 2))
        
        for i, sound in enumerate(self.current_sounds):
            if sound is not None and not self.muted[i]:
                # Apply volume
                volume = self.volumes[i] if not self.soloed[i] else 1.0
                
                # Get the current position in the sound
                pos = self.current_positions[i]
                
                # Get the next chunk of audio
                chunk = sound[pos:pos + frames]
                if len(chunk) < frames:
                    if self.loop_button.isChecked():
                        # Loop the sound
                        chunk = np.tile(chunk, (frames // len(chunk)) + 1)[:frames]
                    else:
                        # Pad with silence
                        chunk = np.pad(chunk, (0, frames - len(chunk)))
                
                # Apply volume and add to mix
                mixed += np.column_stack((chunk, chunk)) * volume
                
                # Update position
                self.current_positions[i] = (pos + frames) % len(sound)
        
        # Store waveform data for display
        self.waveform_data = mixed[:, 0]  # Use left channel for display
        
        # Apply effects
        # TODO: Implement effects processing
        
        # Normalize the output
        if np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed))
            
        outdata[:] = mixed

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SoundMixer()
    window.show()
    sys.exit(app.exec()) 