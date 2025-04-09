import numpy as np
from scipy.io import wavfile
import soundfile as sf

def apply_delay(audio_data, sample_rate, delay_time=0.5, decay=0.5):
    """
    Apply a delay effect to the audio.
    
    Args:
        audio_data (numpy.ndarray): Input audio data
        sample_rate (int): Sample rate of the audio
        delay_time (float): Delay time in seconds
        decay (float): Decay factor for delayed signals (0 to 1)
    
    Returns:
        numpy.ndarray: Processed audio with delay effect
    """
    # Convert delay time to samples
    delay_samples = int(delay_time * sample_rate)
    
    # Create delayed signal
    delayed = np.zeros_like(audio_data)
    delayed[delay_samples:] = audio_data[:-delay_samples] * decay
    
    # Mix original and delayed signals
    output = audio_data + delayed
    
    # Normalize to prevent clipping
    output = output / np.max(np.abs(output))
    
    return output

def apply_reverb(audio_data, sample_rate, room_size=0.8, damping=0.5):
    """
    Apply a simple reverb effect to the audio.
    
    Args:
        audio_data (numpy.ndarray): Input audio data
        sample_rate (int): Sample rate of the audio
        room_size (float): Size of the virtual room (0 to 1)
        damping (float): Damping factor (0 to 1)
    
    Returns:
        numpy.ndarray: Processed audio with reverb effect
    """
    # Create multiple delays with decreasing amplitude
    num_reflections = 8
    output = np.zeros_like(audio_data)
    
    for i in range(num_reflections):
        delay_time = (i + 1) * room_size * 0.1  # Increasing delays
        decay = (1 - damping) ** (i + 1)  # Decreasing amplitudes
        reflection = apply_delay(audio_data, sample_rate, delay_time, decay)
        output += reflection
    
    # Normalize to prevent clipping
    output = output / np.max(np.abs(output))
    
    return output

def process_audio_file(input_file, output_file, effect='delay', **kwargs):
    """
    Process an audio file with the specified effect.
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save processed audio
        effect (str): Effect to apply ('delay' or 'reverb')
        **kwargs: Additional parameters for the effect
    """
    # Read audio file
    audio_data, sample_rate = sf.read(input_file)
    
    # Apply the selected effect
    if effect.lower() == 'delay':
        processed = apply_delay(audio_data, sample_rate, **kwargs)
    elif effect.lower() == 'reverb':
        processed = apply_reverb(audio_data, sample_rate, **kwargs)
    else:
        raise ValueError(f"Unknown effect: {effect}")
    
    # Save processed audio
    sf.write(output_file, processed, sample_rate) 