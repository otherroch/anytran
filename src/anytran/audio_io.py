import os
import sys
import wave

import numpy as np
import soundfile as sf
from moviepy import AudioFileClip
from pydub import AudioSegment


def load_audio_any(input_path):
    _, ext = os.path.splitext(input_path.lower())
    ext = ext[1:] if ext else ''  # Remove leading dot, handle empty extension

    if ext in ("wav", "flac", "ogg"):
        audio, sr = sf.read(input_path)
        return audio, sr

    if ext in ("mp3",):
        audio = AudioSegment.from_mp3(input_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples /= 32768.0
        return samples, audio.frame_rate

    if ext in ("mp4", "mkv", "mov", "m4a"):
        clip = AudioFileClip(input_path)
        try:
            audio = clip.to_soundarray()
            sr = clip.fps

            if audio.ndim > 1:
                audio = audio[:, 0]

            return audio.astype(np.float32), sr
        finally:
            clip.close()

    raise ValueError(f"Unsupported input format: {ext}")


def output_audio(audio_data, output_path=None, play=False):
    if play and audio_data is not None:
        import pyaudio

        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            elif audio_data.dtype != np.int16:
                audio_data = (
                    (audio_data * 32767).astype(np.int16)
                    if np.issubdtype(audio_data.dtype, np.floating)
                    else audio_data.astype(np.int16)
                )
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
        finally:
            p.terminate()
        return

    if output_path is None:
        raise ValueError("output_path must be provided when play=False")

    if not isinstance(audio_data, np.ndarray):
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
    elif audio_data.dtype != np.int16:
        audio_data = (
            (audio_data * 32767).astype(np.int16)
            if np.issubdtype(audio_data.dtype, np.floating)
            else audio_data.astype(np.int16)
        )

    _, ext = os.path.splitext(output_path.lower())
    ext = ext[1:] if ext else ''  # Remove leading dot, handle empty extension

    if ext == "wav":
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data.tobytes())
        if not os.path.exists(output_path):
            raise IOError(f"Failed to create output file: {output_path}")

    elif ext == "mp3":
        try:
            audio = AudioSegment(
                audio_data.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1,
            )
            audio.export(output_path, format="mp3", parameters=["-q:a", "2"])
            if sys.platform != "win32":
                os.sync()
        except Exception as exc:
            raise IOError(f"Failed to create MP3 file: {exc}. Ensure ffmpeg is installed.")
        if not os.path.exists(output_path):
            raise IOError(f"Failed to create output file: {output_path}")
        if os.path.getsize(output_path) == 0:
            raise IOError(f"Output file is empty: {output_path}")

    else:
        raise ValueError(f"Unsupported output format: {ext}")
