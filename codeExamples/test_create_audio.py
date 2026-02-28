import numpy as np
import soundfile as sf

# Create a simple test audio file
duration = 1  # 1 second
sample_rate = 16000
frequency = 440  # A4 note

t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.1 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

sf.write('test_audio.wav', audio, sample_rate)
print("Created test_audio.wav")
