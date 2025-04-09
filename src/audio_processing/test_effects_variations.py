from effects import process_audio_file
import os

def test_effect_variations():
    """
    Test audio effects with different parameters.
    """
    # Input file - using the creek sound
    input_file = "data/processed/sound_samples/gentle-creek-in-rain-forest-with-cicadas.wav"
    
    # Create output directory if it doesn't exist
    output_dir = "data/processed/effects_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different delay parameters
    delay_variations = [
        {
            "name": "short_delay",
            "params": {"delay_time": 0.1, "decay": 0.7}  # Short delay, strong echo
        },
        {
            "name": "long_delay",
            "params": {"delay_time": 0.8, "decay": 0.4}  # Long delay, softer echo
        },
        {
            "name": "multiple_echoes",
            "params": {"delay_time": 0.3, "decay": 0.8}  # Medium delay, very strong echo
        }
    ]
    
    # Process each delay variation
    for variation in delay_variations:
        output_file = os.path.join(output_dir, f"creek_{variation['name']}.wav")
        process_audio_file(
            input_file,
            output_file,
            effect='delay',
            **variation['params']
        )
        print(f"Created {variation['name']} effect: {output_file}")
        print(f"Parameters: {variation['params']}")
    
    # Test different reverb parameters
    reverb_variations = [
        {
            "name": "small_room",
            "params": {"room_size": 0.3, "damping": 0.7}  # Small room, high absorption
        },
        {
            "name": "large_hall",
            "params": {"room_size": 0.9, "damping": 0.2}  # Large space, low absorption
        },
        {
            "name": "medium_space",
            "params": {"room_size": 0.6, "damping": 0.5}  # Medium space, medium absorption
        }
    ]
    
    # Process each reverb variation
    for variation in reverb_variations:
        output_file = os.path.join(output_dir, f"creek_{variation['name']}.wav")
        process_audio_file(
            input_file,
            output_file,
            effect='reverb',
            **variation['params']
        )
        print(f"Created {variation['name']} effect: {output_file}")
        print(f"Parameters: {variation['params']}")

if __name__ == "__main__":
    test_effect_variations() 