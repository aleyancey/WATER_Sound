import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
                           QGroupBox, QDial, QProgressBar)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QFont

class AudioLoader(QThread):
    loaded = pyqtSignal(int, np.ndarray)  # slot, audio_data
    
    def __init__(self, file_path, slot, sample_rate):
        super().__init__()
        self.file_path = file_path
        self.slot = slot
        self.sample_rate = sample_rate
        
    def run(self):
        try:
            # Load audio file using librosa with optimized settings
            audio_data, _ = librosa.load(
                self.file_path,
                sr=self.sample_rate,
                mono=False,
                res_type='kaiser_fast'
            )
            
            # Convert to stereo if mono
            if len(audio_data.shape) == 1:
                audio_data = np.vstack((audio_data, audio_data))
            
            # Transpose and ensure float32
            audio_data = audio_data.T.astype(np.float32)
            
            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.9
                
            self.loaded.emit(self.slot, audio_data)
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            self.loaded.emit(self.slot, None)

class SoundMixer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WATER_Sound - Rain Soundscape Generator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize audio parameters with optimized settings
        self.sample_rate = 44100
        self.buffer_size = 128  # Increased buffer size for better performance
        self.current_sounds = [None] * 2
        self.volumes = [0.5] * 2
        self.muted = [False] * 2
        self.current_positions = [0] * 2
        self.last_mixed_buffer = None
        self.is_playing = False
        self.loop_enabled = True
        self.delay_enabled = False
        self.reverb_enabled = False
        
        # Initialize loading indicators
        self.loading_indicators = [None, None]
        
        # Initialize delay buffer (1 second max delay)
        self.delay_buffer_size = self.sample_rate
        self.delay_buffer = np.zeros((self.delay_buffer_size, 2))
        self.delay_position = 0
        
        # Pre-load sound file paths
        self.sound_file_paths = {}
        self.preload_sound_paths()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create top layout for waveform and sound selections
        top_layout = QHBoxLayout()
        
        # Create left sound selection panel
        left_sound_panel = self.create_sound_selection_panel(0)
        top_layout.addWidget(left_sound_panel)
        
        # Create waveform display
        waveform_panel = self.create_waveform_display()
        top_layout.addWidget(waveform_panel, stretch=2)
        
        # Create right sound selection panel
        right_sound_panel = self.create_sound_selection_panel(1)
        top_layout.addWidget(right_sound_panel)
        
        main_layout.addLayout(top_layout)
        
        # Create effects panel
        self.create_effects_panel(main_layout)
        
        # Create transport controls
        self.create_transport_controls(main_layout)
        
        # Initialize audio stream with optimized settings
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            blocksize=self.buffer_size,
            callback=self.audio_callback,
            finished_callback=self.stream_finished,
            latency='low',
            dtype=np.float32
        )
        
        # Create timer for waveform updates (reduced frequency)
        self.waveform_timer = QTimer()
        self.waveform_timer.timeout.connect(self.update_waveform)
        self.waveform_timer.start(100)  # Update every 100ms instead of 50ms

    def preload_sound_paths(self):
        """Pre-load all sound file paths to improve dropdown responsiveness"""
        sound_types = ["Wood", "Water", "Metal", "Concrete", "Grass", "Leaves", "Heavy rain", "Glass", "City"]
        for sound_type in sound_types:
            sound_dir_name = sound_type.lower().replace(" ", "_")
            sound_dir = os.path.join("data", "raw", "rain_sounds", sound_dir_name)
            if os.path.exists(sound_dir):
                audio_files = [f for f in os.listdir(sound_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
                if audio_files:
                    self.sound_file_paths[sound_type] = os.path.join(sound_dir, audio_files[0])

    def create_sound_selection_panel(self, slot):
        group = QGroupBox(f"Sound {slot + 1}")
        group.setMaximumWidth(100)
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Sound type selection
        combo = QComboBox()
        combo.setMaximumWidth(90)
        combo.addItems(["None", "Test Tone"] + list(self.sound_file_paths.keys()))
        combo.currentIndexChanged.connect(lambda idx: self.sound_selected(slot, combo.currentText()))
        
        # Create a horizontal layout for controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(2)
        
        # Volume slider (vertical)
        volume_slider = QSlider(Qt.Orientation.Vertical)
        volume_slider.setRange(0, 100)
        volume_slider.setValue(50)
        volume_slider.setMaximumWidth(30)
        volume_slider.setMinimumHeight(80)
        volume_slider.setTickPosition(QSlider.TickPosition.TicksRight)
        volume_slider.setTickInterval(10)
        volume_slider.valueChanged.connect(lambda val: self.volume_changed(val, slot))
        
        # Mute button (smaller)
        mute = QPushButton("🔇" if self.muted[slot] else "🔊")
        mute.setCheckable(True)
        mute.setMaximumSize(30, 30)
        mute.toggled.connect(lambda checked: self.mute_toggled(checked, slot))
        
        # Add controls to horizontal layout
        controls_layout.addWidget(volume_slider)
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
        
        # Enable/disable button
        self.delay_toggle = QPushButton("Enable Delay")
        self.delay_toggle.setCheckable(True)
        self.delay_toggle.clicked.connect(self.toggle_delay)
        delay_layout.addWidget(self.delay_toggle)
        
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
        
        # Reverb controls
        reverb_group = QGroupBox("Reverb")
        reverb_layout = QVBoxLayout()
        
        # Enable/disable button
        self.reverb_toggle = QPushButton("Enable Reverb")
        self.reverb_toggle.setCheckable(True)
        self.reverb_toggle.clicked.connect(self.toggle_reverb)
        reverb_layout.addWidget(self.reverb_toggle)
        
        # Room size slider (0-100%)
        self.reverb_size = QSlider(Qt.Orientation.Horizontal)
        self.reverb_size.setRange(0, 100)
        self.reverb_size.setValue(50)
        
        # Damping slider (0-100%)
        self.reverb_damping = QSlider(Qt.Orientation.Horizontal)
        self.reverb_damping.setRange(0, 100)
        self.reverb_damping.setValue(50)
        
        # Mix slider (0-100%)
        self.reverb_mix = QSlider(Qt.Orientation.Horizontal)
        self.reverb_mix.setRange(0, 100)
        self.reverb_mix.setValue(50)
        
        reverb_layout.addWidget(QLabel("Room Size"))
        reverb_layout.addWidget(self.reverb_size)
        reverb_layout.addWidget(QLabel("Damping"))
        reverb_layout.addWidget(self.reverb_damping)
        reverb_layout.addWidget(QLabel("Mix"))
        reverb_layout.addWidget(self.reverb_mix)
        reverb_group.setLayout(reverb_layout)
        
        layout.addWidget(delay_group)
        layout.addWidget(reverb_group)
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
        self.play_button.setCheckable(True)  # Make button checkable
        self.play_button.clicked.connect(self.toggle_playback)
        
        # Add play button to its container
        play_layout.addWidget(self.play_button, alignment=Qt.AlignmentFlag.AlignCenter)
        play_container.setLayout(play_layout)
        
        # Add the play container to the main layout
        layout.addWidget(play_container, stretch=1)  # This will center the play button
        
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def sound_selected(self, slot, sound_type):
        try:
            if sound_type == "Test Tone":
                # Generate test tone immediately
                duration = 2.0
                t = np.linspace(0, duration, int(self.sample_rate * duration), False)
                test_tone = np.sin(2 * np.pi * 440 * t)
                test_tone = (test_tone * 0.3).astype(np.float32)
                test_tone = np.vstack((test_tone, test_tone)).T
                self.current_sounds[slot] = test_tone
                self.current_positions[slot] = 0
            elif sound_type == "None":
                self.current_sounds[slot] = None
                self.current_positions[slot] = 0
            else:
                # Use pre-loaded file path
                file_path = self.sound_file_paths.get(sound_type)
                if file_path is None:
                    raise FileNotFoundError(f"No sound file found for {sound_type}")
                
                # Load audio file with minimal processing
                audio_data, _ = librosa.load(
                    file_path,
                    sr=self.sample_rate,
                    mono=True  # Load as mono first
                )
                
                # Convert to stereo
                audio_data = np.column_stack((audio_data, audio_data))
                
                # Simple normalization
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val
                
                self.current_sounds[slot] = audio_data
                self.current_positions[slot] = 0
                
        except Exception as e:
            print(f"Error loading sound: {e}")
            self.current_sounds[slot] = None
            self.current_positions[slot] = 0

    def on_audio_loaded(self, slot, audio_data):
        """Handle audio data loaded in background"""
        self.loading_indicators[slot].setVisible(False)
        
        if audio_data is not None:
            self.current_sounds[slot] = audio_data
            self.current_positions[slot] = 0
            
            # If playing, restart stream to apply changes immediately
            if self.is_playing:
                self.stream.stop()
                self.stream.start()

    def volume_changed(self, value, slot):
        self.volumes[slot] = value / 100.0
        # No need to restart stream for volume changes
        
    def mute_toggled(self, checked, slot):
        self.muted[slot] = checked
        self.sender().setText("🔇" if checked else "🔊")
        # No need to restart stream for mute changes

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
                # Reset positions and start stream
                for i in range(len(self.current_positions)):
                    self.current_positions[i] = 0
                self.stream.start()
                self.is_playing = True
                self.play_button.setChecked(True)
                self.play_button.setText("⏸")
            else:
                self.stream.stop()
                self.is_playing = False
                self.play_button.setChecked(False)
                self.play_button.setText("▶")
        except sd.PortAudioError as e:
            print(f"Audio stream error: {e}")
            self.play_button.setChecked(False)
            self.is_playing = False
            self.play_button.setText("▶")

    def closeEvent(self, event):
        """Clean up resources when window is closed"""
        if hasattr(self, 'loader') and self.loader.isRunning():
            self.loader.quit()
            self.loader.wait()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        event.accept()

    def update_waveform(self):
        if self.last_mixed_buffer is not None:
            try:
                # Use a smaller buffer for display
                display_size = min(self.buffer_size, len(self.last_mixed_buffer))
                buffer = self.last_mixed_buffer[:display_size, 0]  # Use left channel
                
                # Simple scaling
                if np.max(np.abs(buffer)) > 0:
                    buffer = buffer / np.max(np.abs(buffer))
                
                # Update the waveform display
                self.line.set_ydata(buffer)
                self.canvas.draw()
            except Exception as e:
                print(f"Error updating waveform: {e}")

    def apply_delay(self, audio):
        if not self.delay_enabled or not self.is_playing:
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
            delayed_signal = input_signal + delayed * feedback
            
            # Update delay buffer
            for i in range(len(input_signal)):
                pos = (self.delay_position + i) % self.delay_buffer_size
                self.delay_buffer[pos, channel] = delayed_signal[i]
            
            # Apply mix parameter correctly
            if mix == 0:
                # At 0% mix, return only the original signal
                output[:, channel] = input_signal
            elif mix == 1:
                # At 100% mix, return only the delayed signal
                output[:, channel] = delayed_signal
            else:
                # For values between 0 and 1, mix the signals
                output[:, channel] = input_signal * (1 - mix) + delayed_signal * mix
        
        # Update buffer position
        self.delay_position = (self.delay_position + len(audio)) % self.delay_buffer_size
        
        # Normalize only if necessary to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output

    def apply_reverb(self, audio):
        if not self.reverb_enabled or not self.is_playing:
            return audio
            
        # Calculate reverb parameters
        room_size = self.reverb_size.value() / 100.0
        damping = self.reverb_damping.value() / 100.0
        mix = self.reverb_mix.value() / 100.0
        
        # Pre-calculate delay times and gains based on room size
        base_delays = [0.03, 0.05, 0.07, 0.11, 0.13, 0.17, 0.19, 0.23]
        delay_times = [int(room_size * self.sample_rate * t) for t in base_delays]
        gains = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]  # Reduced gains
        
        # Process each channel
        output = np.zeros_like(audio)
        for channel in range(2):
            input_signal = audio[:, channel]
            channel_output = np.zeros_like(input_signal)
            
            # Process all delay lines at once for better performance
            for delay, gain in zip(delay_times, gains):
                if delay > 0:  # Skip if delay is 0
                    delayed = np.roll(input_signal, delay)
                    delayed[:delay] = 0
                    channel_output += delayed * (1 - damping) * gain
            
            # Normalize and apply feedback
            channel_output = channel_output / len(delay_times)
            feedback = 0.3 * (1 - damping)  # Reduced feedback
            reverbed_signal = input_signal + channel_output * feedback
            
            # Apply mix parameter correctly
            if mix == 0:
                # At 0% mix, return only the original signal
                output[:, channel] = input_signal
            elif mix == 1:
                # At 100% mix, return only the reverbed signal
                output[:, channel] = reverbed_signal
            else:
                # For values between 0 and 1, mix the signals
                output[:, channel] = input_signal * (1 - mix) + reverbed_signal * mix
        
        # Normalize to prevent clipping while preserving original volume
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output

    def audio_callback(self, outdata, frames, time, status):
        try:
            if status:
                print(f"Audio callback status: {status}")
            
            # Initialize output buffer
            outdata.fill(0)
            
            # Mix all active sounds
            for i, sound in enumerate(self.current_sounds):
                if sound is not None and not self.muted[i]:
                    # Get the volume for this sound
                    current_volume = self.volumes[i]
                    
                    # Handle the case where we need to loop during this buffer
                    samples_needed = frames
                    output_position = 0
                    
                    while samples_needed > 0:
                        # Calculate remaining samples in current loop iteration
                        remaining_samples = len(sound) - self.current_positions[i]
                        samples_to_copy = min(samples_needed, remaining_samples)
                        
                        if samples_to_copy > 0:
                            # Add samples to output buffer with volume applied
                            end_pos = self.current_positions[i] + samples_to_copy
                            outdata[output_position:output_position + samples_to_copy] += (
                                sound[self.current_positions[i]:end_pos] * current_volume
                            )
                            
                            self.current_positions[i] += samples_to_copy
                            output_position += samples_to_copy
                            samples_needed -= samples_to_copy
                        
                        # Check if we need to loop
                        if self.current_positions[i] >= len(sound):
                            if self.loop_enabled:
                                self.current_positions[i] = 0
                            else:
                                # If not looping, stop this sound
                                self.current_sounds[i] = None
                                self.current_positions[i] = 0
                                break
            
            # Apply effects
            if self.delay_enabled:
                outdata[:] = self.apply_delay(outdata)
            
            if self.reverb_enabled:
                outdata[:] = self.apply_reverb(outdata)
            
            # Store the mixed buffer for waveform display
            self.last_mixed_buffer = outdata.copy()
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)

    def toggle_delay(self, checked):
        """Toggle delay effect on/off"""
        self.delay_enabled = checked
        self.delay_toggle.setText("Disable Delay" if checked else "Enable Delay")
        if checked:
            self.reset_delay_buffer()

    def toggle_reverb(self, checked):
        """Toggle reverb effect on/off"""
        self.reverb_enabled = checked
        self.reverb_toggle.setText("Disable Reverb" if checked else "Enable Reverb")
        # No need to restart stream, effect will be applied in next callback

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SoundMixer()
    window.show()
    sys.exit(app.exec()) 