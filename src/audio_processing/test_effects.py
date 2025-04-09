from effects import process_audio_file
import os

def test_effects():
    """
    Test the audio effects on a sample file.
    """
    # Input file - using one of the water sound samples
    input_file = "data/processed/sound_samples/gentle-creek-in-rain-forest-with-cicadas.wav"
    
    # Create output directory if it doesn't exist
    output_dir = "data/processed/effects_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test delay effect
    delay_output = os.path.join(output_dir, "creek_with_delay.wav")
    process_audio_file(
        input_file,
        delay_output,
        effect='delay',
        delay_time=0.3,  # 300ms delay
        decay=0.6
    )
    print(f"Created delay effect: {delay_output}")
    
    # Test reverb effect
    reverb_output = os.path.join(output_dir, "creek_with_reverb.wav")
    process_audio_file(
        input_file,
        reverb_output,
        effect='reverb',
        room_size=0.7,
        damping=0.4
    )
    print(f"Created reverb effect: {reverb_output}")

if __name__ == "__main__":
    test_effects() 