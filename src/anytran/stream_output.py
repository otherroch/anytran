import sys

import librosa
import numpy as np


def get_wasapi_loopback_device_info(preferred_name=None, verbose=False):
    if sys.platform != "win32":
        return None

    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        import pyaudio

    p = pyaudio.PyAudio()
    try:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            wasapi_index = wasapi_info.get("index")
        except Exception:
            wasapi_index = None

        default_output = None
        try:
            default_output = p.get_default_output_device_info()
        except Exception:
            default_output = None

        loopback_devices = []
        for idx in range(p.get_device_count()):
            info = p.get_device_info_by_index(idx)
            if wasapi_index is not None and info.get("hostApi") != wasapi_index:
                continue
            if info.get("maxInputChannels", 0) <= 0:
                continue
            name = info.get("name", "")
            is_loopback = bool(info.get("isLoopbackDevice")) or "loopback" in name.lower()
            if not is_loopback:
                continue
            loopback_devices.append(info)

        if not loopback_devices:
            return None

        if preferred_name:
            preferred_lower = preferred_name.lower()
            for info in loopback_devices:
                if preferred_lower in info.get("name", "").lower():
                    return info

        if default_output:
            default_name = default_output.get("name", "")
            default_name_lower = default_name.lower()
            for info in loopback_devices:
                name_lower = info.get("name", "").lower()
                if default_name_lower and default_name_lower in name_lower:
                    return info

        if verbose:
            print("Loopback devices found:")
            for info in loopback_devices:
                print(f"  {info.get('index')}: {info.get('name')}")

        return loopback_devices[0]
    finally:
        p.terminate()


def list_wasapi_loopback_devices():
    if sys.platform != "win32":
        print("System output capture is only supported on Windows (WASAPI loopback).")
        return

    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        import pyaudio

    p = pyaudio.PyAudio()
    try:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            wasapi_index = wasapi_info.get("index")
        except Exception:
            wasapi_index = None

        default_output = None
        try:
            default_output = p.get_default_output_device_info()
        except Exception:
            default_output = None

        loopback_devices = []
        for idx in range(p.get_device_count()):
            info = p.get_device_info_by_index(idx)
            if wasapi_index is not None and info.get("hostApi") != wasapi_index:
                continue
            if info.get("maxInputChannels", 0) <= 0:
                continue
            name = info.get("name", "")
            is_loopback = bool(info.get("isLoopbackDevice")) or "loopback" in name.lower()
            if not is_loopback:
                continue
            loopback_devices.append(info)

        if not loopback_devices:
            print("No WASAPI loopback devices found.")
            return

        default_name = default_output.get("name", "") if default_output else ""
        print("WASAPI loopback devices:")
        for info in loopback_devices:
            name = info.get("name", "")
            marker = " (default)" if default_name and default_name.lower() in name.lower() else ""
            print(f"  {info.get('index')}: {name}{marker}")
    finally:
        p.terminate()


def stream_output_audio(audio_queue, device_info, target_rate=16000, stop_event=None, verbose=False):
    if sys.platform != "win32":
        print("System output capture is only supported on Windows (WASAPI loopback).")
        return

    if not device_info:
        print("No WASAPI loopback device found. Cannot capture system output audio.")
        return

    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        import pyaudio

    p = pyaudio.PyAudio()
    stream = None
    try:
        device_index = device_info.get("index")
        device_rate = int(device_info.get("defaultSampleRate", target_rate))
        channels = int(device_info.get("maxInputChannels", 1))
        if channels < 1:
            channels = 1
        if channels > 2:
            channels = 2

        frames_per_buffer = max(256, int(device_rate / 10))

        if verbose:
            print(f"Capturing output device: {device_info.get('name')}")
            print(f"Device rate: {device_rate} Hz, channels: {channels}")

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=device_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frames_per_buffer,
        )

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio_np = audio_np.reshape(-1, channels).mean(axis=1)

            if device_rate != target_rate:
                audio_np = librosa.resample(audio_np, orig_sr=device_rate, target_sr=target_rate)

            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except Exception:
                    pass
            audio_queue.put(audio_np)
    except Exception as exc:
        print(f"Error capturing system output audio: {exc}")
    finally:
        try:
            if stream:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        p.terminate()
