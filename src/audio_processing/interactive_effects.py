from effects import process_audio_file
import os

def get_float_input(prompt, min_val=0.0, max_val=1.0):
    """Get float input from user with validation."""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def apply_interactive_effects():
    """
    Interactive function to apply audio effects with user-defined parameters.
    """
    # Input file selection
    print("\nAvailable input files:")
    input_files = {
        "1": "data/processed/sound_samples/gentle-creek-in-rain-forest-with-cicadas.wav",
        "2": "data/processed/sound_samples/_gentle-rain.wav",
        "3": "data/processed/sound_samples/1982_gentle-rain-on-leaves-with-soft-wind-and-suburban-ambience.wav"
    }
    
    for key, path in input_files.items():
        print(f"{key}: {os.path.basename(path)}")
    
    while True:
        file_choice = input("\nSelect input file number (1-3): ")
        if file_choice in input_files:
            input_file = input_files[file_choice]
            break
        print("Please select a valid file number")

    # Create output directory
    output_dir = "data/processed/effects_test"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        # Effect selection
        print("\nAvailable effects:")
        print("1: Delay")
        print("2: Reverb")
        print("3: Exit")
        
        effect_choice = input("\nSelect effect (1-3): ")
        
        if effect_choice == "3":
            break
            
        if effect_choice == "1":
            # Delay effect parameters
            print("\n=== Delay Effect Parameters ===")
            delay_time = get_float_input("Enter delay time (0.1-2.0 seconds): ", 0.1, 2.0)
            decay = get_float_input("Enter decay (0.0-1.0): ")
            
            output_file = os.path.join(output_dir, f"custom_delay_{delay_time}s_{decay}decay.wav")
            process_audio_file(
                input_file,
                output_file,
                effect='delay',
                delay_time=delay_time,
                decay=decay
            )
            print(f"\nCreated delay effect: {output_file}")
            print(f"Parameters: delay_time={delay_time}, decay={decay}")
            
        elif effect_choice == "2":
            # Reverb effect parameters
            print("\n=== Reverb Effect Parameters ===")
            room_size = get_float_input("Enter room size (0.0-1.0): ")
            damping = get_float_input("Enter damping (0.0-1.0): ")
            
            output_file = os.path.join(output_dir, f"custom_reverb_{room_size}room_{damping}damp.wav")
            process_audio_file(
                input_file,
                output_file,
                effect='reverb',
                room_size=room_size,
                damping=damping
            )
            print(f"\nCreated reverb effect: {output_file}")
            print(f"Parameters: room_size={room_size}, damping={damping}")
        
        # Ask if user wants to continue
        if input("\nWould you like to try another effect? (y/n): ").lower() != 'y':
            break

    print("\nDone! Check the effects_test directory for your processed files.")

if __name__ == "__main__":
    print("=== Interactive Audio Effects ===")
    apply_interactive_effects() 