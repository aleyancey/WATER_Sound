import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
                           QGroupBox, QDial)
from PyQt6.QtCore import Qt, QTimer
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QFont

class SoundMixer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WATER_Sound - Rain Soundscape Generator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize audio parameters
        self.sample_rate = 44100
        self.buffer_size = 1024
        self.current_sounds = [None] * 2
        self.volumes = [1.0] * 2
        self.muted = [False] * 2
        self.current_positions = [0] * 2
        self.last_mixed_buffer = None
        self.is_playing = False
        
        # Initialize delay buffer (2 seconds max delay)
        self.delay_buffer_size = self.sample_rate * 2
        self.delay_buffer = np.zeros((self.delay_buffer_size, 2))
        self.delay_position = 0
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create top layout for waveform and sound selections
        top_layout = QHBoxLayout()
        
        # Create left sound selection panel
        left_sound_panel = self.create_sound_selection_panel(0)
        top_layout.addWidget(left_sound_panel)
        
        # Create waveform display (larger)
        waveform_panel = self.create_waveform_display()
        top_layout.addWidget(waveform_panel, stretch=2)  # Give waveform more space
        
        # Create right sound selection panel
        right_sound_panel = self.create_sound_selection_panel(1)
        top_layout.addWidget(right_sound_panel)
        
        main_layout.addLayout(top_layout)
        
        # Create effects panel
        self.create_effects_panel(main_layout)
        
        # Create transport controls
        self.create_transport_controls(main_layout)
        
        # Initialize audio stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            blocksize=self.buffer_size,
            callback=self.audio_callback,
            finished_callback=self.stream_finished
        )
        
        # Create timer for waveform updates
        self.waveform_timer = QTimer()
        self.waveform_timer.timeout.connect(self.update_waveform)
        self.waveform_timer.start(50)

    def create_sound_selection_panel(self, slot):
        group = QGroupBox(f"Sound {slot + 1}")
        group.setMaximumWidth(100)  # Much narrower panel
        layout = QVBoxLayout()
        layout.setSpacing(5)  # Reduce spacing between elements
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Sound type selection (smaller combo box)
        combo = QComboBox()
        combo.setMaximumWidth(90)  # Even narrower combo box
        combo.addItems(["None", "Test Tone (440Hz)", "Wood", "Water", "Metal", "Concrete", 
                      "Grass", "Leaves", "Heavy rain", "Glass", "Medium rain"])
        combo.currentIndexChanged.connect(lambda idx: self.sound_selected(idx, slot))
        
        # Create a horizontal layout for controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(2)  # Minimal spacing between controls
        
        # Volume dial (smaller)
        volume_dial = QDial()
        volume_dial.setRange(0, 100)
        volume_dial.setValue(100)
        volume_dial.setNotchesVisible(True)
        volume_dial.setWrapping(False)
        volume_dial.setMaximumSize(40, 40)  # Much smaller dial
        volume_dial.valueChanged.connect(lambda val: self.volume_changed(val, slot))
        
        # Mute button (smaller)
        mute = QPushButton("🔇" if self.muted[slot] else "🔊")
        mute.setCheckable(True)
        mute.setMaximumSize(30, 30)  # Smaller button
        mute.toggled.connect(lambda checked: self.mute_toggled(checked, slot))
        
        # Add controls to horizontal layout
        controls_layout.addWidget(volume_dial)
        controls_layout.addWidget(mute)
        
        # Add widgets to main layout
        layout.addWidget(combo, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(controls_layout)
        
        group.setLayout(layout)
        return group
        
    def create_waveform_display(self):
        group = QGroupBox("Waveform Display")
        layout = QVBoxLayout()
        
        # Create matplotlib figure (larger)
        self.figure = Figure(figsize=(20, 4))  # Even wider display
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.buffer_size)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor('#f0f0f0')
        self.figure.patch.set_facecolor('#ffffff')
        
        # Add grid for better visibility
        self.ax.grid(True, linestyle='-', alpha=0.3)
        
        # Create initial line with thicker width
        x = np.arange(self.buffer_size)
        y = np.zeros(self.buffer_size)
        self.line, = self.ax.plot(x, y, color='#1f77b4', linewidth=2)
        
        layout.addWidget(self.canvas)
        group.setLayout(layout)
        return group

    def create_effects_panel(self, parent_layout):
        group = QGroupBox("Effects")
        layout = QHBoxLayout()
        
        # Delay controls
        delay_group = QGroupBox("Delay")
        delay_layout = QVBoxLayout()
        
        # Time slider (0-2000ms)
        self.delay_time = QSlider(Qt.Orientation.Horizontal)
        self.delay_time.setRange(0, 2000)
        self.delay_time.setValue(500)  # Default to 500ms
        self.delay_time.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.delay_time.setTickInterval(500)
        
        # Feedback slider (0-100%)
        self.delay_feedback = QSlider(Qt.Orientation.Horizontal)
        self.delay_feedback.setRange(0, 100)
        self.delay_feedback.setValue(50)  # Default to 50%
        
        # Mix slider (0-100%)
        self.delay_mix = QSlider(Qt.Orientation.Horizontal)
        self.delay_mix.setRange(0, 100)
        self.delay_mix.setValue(50)  # Default to 50%
        
        delay_layout.addWidget(QLabel("Time (ms)"))
        delay_layout.addWidget(self.delay_time)
        delay_layout.addWidget(QLabel("Feedback"))
        delay_layout.addWidget(self.delay_feedback)
        delay_layout.addWidget(QLabel("Mix"))
        delay_layout.addWidget(self.delay_mix)
        delay_group.setLayout(delay_layout)
        
        layout.addWidget(delay_group)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_transport_controls(self, parent_layout):
        group = QGroupBox("Transport Controls")
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        # Create a container for the play button to center it
        play_container = QWidget()
        play_layout = QHBoxLayout()
        play_layout.setContentsMargins(0, 0, 0, 0)
        
        # Play button (larger, centered)
        self.play_button = QPushButton("▶")
        self.play_button.setFont(QFont("Arial", 20))
        self.play_button.setFixedSize(60, 60)
        self.play_button.clicked.connect(self.toggle_playback)
        
        # Add play button to its container
        play_layout.addWidget(self.play_button, alignment=Qt.AlignmentFlag.AlignCenter)
        play_container.setLayout(play_layout)
        
        # Add the play container to the main layout
        layout.addWidget(play_container, stretch=1)  # This will center the play button
        
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def sound_selected(self, index, slot):
        if index == 0:  # None
            self.current_sounds[slot] = None
            return
            
        if index == 1:  # Test Tone
            # Generate a 440Hz sine wave
            duration = 5  # seconds
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            self.current_sounds[slot] = 0.5 * np.sin(2 * np.pi * 440 * t)
        else:
            # Load the selected sound
            sound_type = self.sender().currentText().lower().replace(" ", "_")
            sound_path = f"data/raw/rain_sounds/{sound_type}/"
            if os.path.exists(sound_path):
                files = [f for f in os.listdir(sound_path) if f.endswith(('.wav', '.mp3'))]
                if files:
                    self.current_sounds[slot] = librosa.load(
                        os.path.join(sound_path, files[0]),
                        sr=self.sample_rate
                    )[0]
        
    def volume_changed(self, value, slot):
        # Only update volume if not muted
        if not self.muted[slot]:
            self.volumes[slot] = value / 100.0
        
    def mute_toggled(self, checked, slot):
        self.muted[slot] = checked
        # Update volume immediately when mute is toggled
        if checked:
            self.volumes[slot] = 0
            # Update mute button text
            self.sender().setText("🔇")
        else:
            # Restore previous volume
            volume_dial = self.sender().parent().findChild(QDial)
            if volume_dial:
                self.volumes[slot] = volume_dial.value() / 100.0
            # Update mute button text
            self.sender().setText("🔊")
        
    def reset_delay_buffer(self):
        """Reset the delay buffer to zeros"""
        self.delay_buffer.fill(0)
        self.delay_position = 0

    def stream_finished(self):
        """Callback when stream is finished"""
        self.is_playing = False
        self.play_button.setChecked(False)
        self.reset_delay_buffer()

    def toggle_playback(self):
        try:
            if not self.is_playing:
                self.reset_delay_buffer()
                self.stream.start()
                self.is_playing = True
            else:
                self.stream.stop()
                self.is_playing = False
                self.reset_delay_buffer()
        except sd.PortAudioError as e:
            print(f"Audio stream error: {e}")
            self.play_button.setChecked(False)
            self.is_playing = False

    def closeEvent(self, event):
        """Clean up resources when window is closed"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        event.accept()

    def update_waveform(self):
        if self.last_mixed_buffer is not None:
            # Ensure the buffer size matches
            buffer = self.last_mixed_buffer[:self.buffer_size, 0]  # Use left channel
            if len(buffer) < self.buffer_size:
                buffer = np.pad(buffer, (0, self.buffer_size - len(buffer)))
            
            # Scale the waveform for better visibility
            if np.max(np.abs(buffer)) > 0:
                buffer = buffer / np.max(np.abs(buffer))
            
            # Update the waveform display
            self.line.set_ydata(buffer)
            self.canvas.draw()
            
    def apply_delay(self, audio):
        if self.delay_mix.value() == 0 or not self.is_playing:
            return audio
            
        # Calculate delay parameters
        delay_samples = int(self.delay_time.value() * self.sample_rate / 1000)
        feedback = self.delay_feedback.value() / 100.0
        mix = self.delay_mix.value() / 100.0
        
        # Process each channel
        output = np.zeros_like(audio)
        for channel in range(2):
            # Get current input
            input_signal = audio[:, channel]
            
            # Calculate read position (current position - delay)
            read_pos = (self.delay_position - delay_samples) % self.delay_buffer_size
            
            # Get delayed signal
            delayed = np.zeros_like(input_signal)
            for i in range(len(input_signal)):
                pos = (read_pos + i) % self.delay_buffer_size
                delayed[i] = self.delay_buffer[pos, channel]
            
            # Mix input with feedback
            output[:, channel] = input_signal + delayed * feedback
            
            # Update delay buffer
            for i in range(len(input_signal)):
                pos = (self.delay_position + i) % self.delay_buffer_size
                self.delay_buffer[pos, channel] = output[i, channel]
            
        # Update buffer position
        self.delay_position = (self.delay_position + len(audio)) % self.delay_buffer_size
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val
        
        # Mix with original signal
        return audio * (1 - mix) + output * mix

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
            
        try:
            # Mix all active sounds
            mixed = np.zeros((frames, 2))
            
            for i, sound in enumerate(self.current_sounds):
                if sound is not None and not self.muted[i]:
                    # Apply volume
                    volume = self.volumes[i]
                    
                    # Get the current position in the sound
                    pos = self.current_positions[i]
                    
                    # Get the next chunk of audio
                    if pos < len(sound):
                        chunk = sound[pos:pos + frames]
                        if len(chunk) > 0:  # Only process if chunk is not empty
                            if len(chunk) < frames:
                                # Always loop the sound for continuous playback
                                repeats = (frames // len(chunk)) + 1
                                chunk = np.tile(chunk, repeats)[:frames]
                            
                            # Apply volume and add to mix
                            mixed += np.column_stack((chunk, chunk)) * volume
                            
                            # Update position
                            self.current_positions[i] = (pos + frames) % len(sound)
            
            # Store the mixed buffer for visualization
            self.last_mixed_buffer = mixed.copy()
            
            # Apply delay effect
            if self.is_playing:
                mixed = self.apply_delay(mixed)
            
            # Normalize the output
            max_val = np.max(np.abs(mixed))
            if max_val > 0:
                mixed = mixed / max_val
                
            outdata[:] = mixed
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SoundMixer()
    window.show()
    sys.exit(app.exec()) 