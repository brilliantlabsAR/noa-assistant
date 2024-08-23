import webrtcvad
import wave
import numpy as np
from pydub import AudioSegment
from timeit import default_timer as timer

def timeit(func):
    """Decorator to measure time taken by a function."""
    def wrapper(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def read_wave(path):
    """Reads a .wav file and ensures it's in the correct format for VAD processing."""
    audio = AudioSegment.from_file(path)

    # Convert to mono and 16-bit PCM
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    pcm_data = np.array(audio.get_array_of_samples(), dtype=np.int16)

    # Convert to raw bytes for VAD
    pcm_data = pcm_data.tobytes()
    sample_rate = audio.frame_rate

    return pcm_data, sample_rate

def read_from_bytes(data, sample_rate):
    """Reads PCM audio data from bytes and ensures it's in the correct format for VAD processing."""
    pcm_data = np.frombuffer(data, dtype=np.int16)

    # Convert to raw bytes for VAD
    pcm_data = pcm_data.tobytes()

    return pcm_data, sample_rate
def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data."""
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # frame size in bytes
    num_frames = len(audio) // frame_size
    for i in range(num_frames):
        yield audio[i * frame_size:(i + 1) * frame_size]

@timeit
def is_speech_present(audio_bytes, aggressiveness=3, threshold=0.1):
    """Checks if there is any speech present in the WAV file using VAD."""
    audio, sample_rate = read_from_bytes(audio_bytes, 16000)

    vad = webrtcvad.Vad(aggressiveness)  # Aggressiveness mode (0-3)

    frame_duration_ms = 10  # Reduce duration of each frame to 10 ms
    frames = list(frame_generator(frame_duration_ms, audio, sample_rate))
    
    # Process frames in batch for speed
    num_voiced_frames = sum(1 for frame in frames if vad.is_speech(frame, sample_rate))
    total_frames = len(frames)

    # Speech is detected if more than a threshold percentage of frames are voiced
    speech_detected = num_voiced_frames > threshold * total_frames

    return speech_detected

if __name__ == "__main__":
    audio_path = "audio6.wav"  # Path to your input .wav file
    audio_bytes, sample_rate = read_wave(audio_path)
    
    if is_speech_present(audio_bytes=audio_bytes):
        print("Speech detected!")
    else:
        print("No speech detected.")
