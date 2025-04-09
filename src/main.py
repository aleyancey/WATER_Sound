import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
                           QGroupBox)
from PyQt6.QtCore import Qt, QTimer
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SoundMixer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WATER_Sound - Rain Soundscape Generator")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize audio parameters
        self.sample_rate = 44100
        self.buffer_size = 1024
        self.current_sounds = [None] * 4
        self.volumes = [1.0] * 4
        self.muted = [False] * 4
        self.current_positions = [0] * 4
        self.waveform_data = np.zeros(self.buffer_size)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create sound selection panel
        self.create_sound_selection_panel(layout)
        
        # Create waveform display
        self.create_waveform_display(layout)
        
        # Create transport controls
        self.create_transport_controls(layout)
        
        # Initialize audio stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=self.audio_callback,
            blocksize=self.buffer_size
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
            volume_layout = QVBoxLayout()
            volume_label = QLabel("Volume: 100%")
            volume = QSlider(Qt.Orientation.Horizontal)
            volume.setRange(0, 100)
            volume.setValue(100)
            volume.valueChanged.connect(lambda val, slot=i, label=volume_label: 
                                     self.volume_changed(val, slot, label))
            volume_layout.addWidget(volume_label)
            volume_layout.addWidget(volume)
            
            # Mute button
            mute = QPushButton("Mute")
            mute.setCheckable(True)
            mute.toggled.connect(lambda checked, slot=i: self.mute_toggled(checked, slot))
            
            slot_layout.addWidget(combo, stretch=2)
            slot_layout.addLayout(volume_layout, stretch=2)
            slot_layout.addWidget(mute)
            
            # Add a status label
            status_label = QLabel("")
            slot_layout.addWidget(status_label)
            setattr(self, f'status_label_{i}', status_label)
            
            layout.addLayout(slot_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_waveform_display(self, parent_layout):
        group = QGroupBox("Waveform Display")
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(6, 2))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.buffer_size)
        self.ax.grid(True)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.waveform_line, = self.ax.plot(np.arange(self.buffer_size), np.zeros(self.buffer_size))
        
        layout.addWidget(self.canvas)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_transport_controls(self, parent_layout):
        group = QGroupBox("Transport Controls")
        layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self.play_toggled)
        
        self.loop_button = QPushButton("Loop")
        self.loop_button.setCheckable(True)
        
        layout.addWidget(self.play_button)
        layout.addWidget(self.loop_button)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def sound_selected(self, index, slot):
        status_label = getattr(self, f'status_label_{slot}')
        if index == 0:
            self.current_sounds[slot] = None
            status_label.setText("")
            return
            
        # Load the selected sound
        sound_type = self.sender().currentText().lower().replace(" ", "_")
        try:
            sound_path = f"data/raw/rain_sounds/{sound_type}/"
            if not os.path.exists(sound_path):
                status_label.setText("❌ Directory not found")
                return
                
            files = [f for f in os.listdir(sound_path) if f.endswith(('.wav', '.mp3'))]
            if not files:
                status_label.setText("❌ No audio files found")
                return
                
            file_path = os.path.join(sound_path, files[0])
            self.current_sounds[slot] = librosa.load(file_path, sr=self.sample_rate)[0]
            status_label.setText("✓ Loaded")
            print(f"Loaded sound: {file_path}")
            
        except Exception as e:
            print(f"Error loading sound: {e}")
            status_label.setText("❌ Error loading")
            self.current_sounds[slot] = None
                
    def volume_changed(self, value, slot, label):
        self.volumes[slot] = value / 100.0
        label.setText(f"Volume: {value}%")
        
    def mute_toggled(self, checked, slot):
        self.muted[slot] = checked
        status_label = getattr(self, f'status_label_{slot}')
        if checked:
            status_label.setText("🔇 Muted")
        elif self.current_sounds[slot] is not None:
            status_label.setText("✓ Loaded")
        
    def play_toggled(self, checked):
        if checked:
            self.stream.start()
            self.play_button.setText("Stop")
        else:
            self.stream.stop()
            self.play_button.setText("Play")
            
    def update_waveform(self):
        if hasattr(self, 'waveform_data'):
            if len(self.waveform_data) != self.buffer_size:
                self.waveform_data = np.interp(
                    np.linspace(0, len(self.waveform_data) - 1, self.buffer_size),
                    np.arange(len(self.waveform_data)),
                    self.waveform_data
                )
            self.waveform_line.set_ydata(self.waveform_data)
            self.canvas.draw()
            
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
            
        # Mix all active sounds
        mixed = np.zeros((frames, 2))
        
        for i, sound in enumerate(self.current_sounds):
            if sound is not None and not self.muted[i]:
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
                mixed += np.column_stack((chunk, chunk)) * self.volumes[i]
                
                # Update position
                self.current_positions[i] = (pos + frames) % len(sound)
        
        # Store waveform data for display
        self.waveform_data = mixed[:, 0]
        
        # Normalize the output to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.9  # Leave some headroom
            
        outdata[:] = mixed

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SoundMixer()
    window.show()
    sys.exit(app.exec()) 