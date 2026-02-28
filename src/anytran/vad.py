try:
    from silero_vad import load_silero_vad, get_speech_timestamps

    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    print("Silero VAD not available. Using magnitude threshold for voice detection.")
    print("Install with: pip install silero-vad")

_vad_model = None


def get_vad_model():
    global _vad_model
    if _vad_model is None and SILERO_AVAILABLE:
        _vad_model = load_silero_vad()
    return _vad_model


def has_speech_silero(audio_data, sample_rate=16000, min_speech_ms=300, min_silence_ms=200):
    if not SILERO_AVAILABLE:
        return None

    try:
        vad = get_vad_model()
        if vad is None:
            return None

        speech_timestamps = get_speech_timestamps(
            audio_data,
            vad,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            return_seconds=False,
        )
        return len(speech_timestamps) > 0
    except Exception as exc:
        print(f"VAD error: {exc}")
        return None
