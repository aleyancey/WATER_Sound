import sounddevice as sd
import numpy as np
import librosa
import time

def test_loop():
    # Load a short sound (metal/electric passenger train)
    sound_path = "data/raw/rain_sounds/metal/electric passenger train passing MS 141124_0464.wav"
    audio, sr = librosa.load(sound_path, sr=44100)
    
    # Initialize audio parameters
    buffer_size = 1024
    position = 0
    
    def audio_callback(outdata, frames, time, status):
        nonlocal position
        if status:
            print(status)
            
        # Get the next chunk
        chunk = audio[position:position + frames]
        if len(chunk) < frames:
            # Loop the sound
            repeats = (frames // len(chunk)) + 1
            chunk = np.tile(chunk, repeats)[:frames]
        
        # Update position
        position = (position + frames) % len(audio)
        
        # Output the chunk
        outdata[:] = np.column_stack((chunk, chunk))
    
    # Create and start the stream
    stream = sd.OutputStream(
        samplerate=sr,
        channels=2,
        blocksize=buffer_size,
        callback=audio_callback
    )
    
    print("Playing sound with loop...")
    print("Press Ctrl+C to stop")
    
    try:
        stream.start()
        # Play for 10 seconds to test the loop
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        print("Stopped playback")

if __name__ == "__main__":
    test_loop() 