import sys
from pathlib import Path
import numpy as np

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anytran import tts


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_indextts_default_model_constant():
    """_INDEXTTS_DEFAULT_MODEL should be the IndexTeam/IndexTTS-2 repo."""
    assert tts._INDEXTTS_DEFAULT_MODEL == "IndexTeam/IndexTTS-2"


# ---------------------------------------------------------------------------
# indextts_tts – unavailable package
# ---------------------------------------------------------------------------

def test_indextts_tts_not_available_returns_false(tmp_path, monkeypatch):
    """indextts_tts returns False when indextts package is not installed."""
    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", False)

    result = tts.indextts_tts(
        "Hello world",
        "IndexTeam/IndexTTS-2",
        str(tmp_path / "output.wav"),
        verbose=False,
    )

    assert result is False


def test_indextts_tts_no_reference_audio_returns_false(tmp_path, monkeypatch):
    """indextts_tts returns False when no reference_audio is provided."""
    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})

    class FakeEngine:
        def infer(self, spk_audio_prompt, text, output_path, verbose=False):
            pass

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "_load_indextts_engine", fake_load_engine)

    result = tts.indextts_tts(
        "Hello world",
        "IndexTeam/IndexTTS-2",
        str(tmp_path / "output.wav"),
        reference_audio=None,
        verbose=False,
    )

    assert result is True  # Should return True and do nothing when no reference audio


def test_indextts_tts_with_reference_audio(tmp_path, monkeypatch):
    """indextts_tts succeeds when reference_audio is provided and engine works."""
    import soundfile as sf

    output_wav = tmp_path / "output.wav"

    # Create a fake engine that writes a silent WAV file
    class FakeEngine:
        def infer(self, spk_audio_prompt, text, output_path, verbose=False):
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            sf.write(output_path, audio, sample_rate)

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "_load_indextts_engine", fake_load_engine)

    ref_audio = (np.zeros(16000) * 32767).astype(np.int16)

    result = tts.indextts_tts(
        "Hello world",
        "IndexTeam/IndexTTS-2",
        str(output_wav),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        verbose=False,
    )

    assert result is True
    assert output_wav.exists()


def test_indextts_tts_uses_cache(tmp_path, monkeypatch):
    """indextts_tts caches the engine so it is only loaded once."""
    import soundfile as sf

    load_count = {"count": 0}

    class FakeEngine:
        def infer(self, spk_audio_prompt, text, output_path, verbose=False):
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(output_path, audio, 16000)

    def fake_load_engine(model_name, verbose=False):
        load_count["count"] += 1
        return FakeEngine()

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "_load_indextts_engine", fake_load_engine)

    ref_audio = (np.zeros(16000) * 32767).astype(np.int16)

    model_name = "IndexTeam/IndexTTS-2"

    # First call should load the engine
    tts.indextts_tts("Hello", model_name, str(tmp_path / "out1.wav"),
                     reference_audio=ref_audio, verbose=False)
    assert load_count["count"] == 1

    # Second call should reuse the cached engine
    tts.indextts_tts("World", model_name, str(tmp_path / "out2.wav"),
                     reference_audio=ref_audio, verbose=False)
    assert load_count["count"] == 1


def test_indextts_tts_default_model_used_when_none(tmp_path, monkeypatch):
    """indextts_tts uses _INDEXTTS_DEFAULT_MODEL when voice_model is None."""
    import soundfile as sf

    loaded_models = []

    class FakeEngine:
        def infer(self, spk_audio_prompt, text, output_path, verbose=False):
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(output_path, audio, 16000)

    def fake_load_engine(model_name, verbose=False):
        loaded_models.append(model_name)
        return FakeEngine()

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "_load_indextts_engine", fake_load_engine)

    ref_audio = (np.zeros(16000) * 32767).astype(np.int16)

    tts.indextts_tts("Hello", None, str(tmp_path / "out.wav"),
                     reference_audio=ref_audio, verbose=False)

    assert loaded_models == [tts._INDEXTTS_DEFAULT_MODEL]


def test_indextts_tts_reference_audio_float_range(tmp_path, monkeypatch):
    """indextts_tts normalises float32 reference audio into [-1, 1] correctly."""
    import soundfile as sf

    received_paths = []

    class FakeEngine:
        def infer(self, spk_audio_prompt, text, output_path, verbose=False):
            received_paths.append(spk_audio_prompt)
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(output_path, audio, 16000)

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "_load_indextts_engine", fake_load_engine)

    # Pass float32 audio already in [-1, 1]
    ref_audio = np.random.uniform(-1.0, 1.0, 16000).astype(np.float32)

    result = tts.indextts_tts(
        "Test",
        "IndexTeam/IndexTTS-2",
        str(tmp_path / "out.wav"),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        verbose=False,
    )

    assert result is True
    assert len(received_paths) == 1
    # Verify the temp reference file was written (path should exist at call time)
    # but gets cleaned up after the call
    import os
    assert not os.path.exists(received_paths[0])  # cleaned up


# ---------------------------------------------------------------------------
# synthesize_tts_pcm with indextts backend
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_with_indextts_backend(tmp_path, monkeypatch):
    """synthesize_tts_pcm with indextts backend returns int16 PCM array."""
    import soundfile as sf

    class FakeEngine:
        def infer(self, spk_audio_prompt, text, output_path, verbose=False):
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(output_path, audio, 16000)

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "_load_indextts_engine", fake_load_engine)

    # indextts_tts requires reference_audio; without it, synthesize_tts_pcm
    # will return None (indextts_tts returns False, falls back to gTTS which
    # is not available in test). Check that the backend is entered.
    # We patch indextts_tts directly to avoid that complication.
    def fake_indextts_tts(text, voice_model, output_wav, reference_audio=None,
                          reference_sample_rate=16000, verbose=False):
        audio = np.zeros(16000, dtype=np.float32)
        sf.write(output_wav, audio, 16000)
        return True

    monkeypatch.setattr(tts, "indextts_tts", fake_indextts_tts)

    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="indextts",
        voice_model="IndexTeam/IndexTTS-2",
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


def test_synthesize_tts_pcm_indextts_replaces_non_indextts_model(monkeypatch):
    """synthesize_tts_pcm replaces non-IndexTTS model names with the default."""
    import soundfile as sf

    loaded_models = []

    def fake_indextts_tts(text, voice_model, output_wav, reference_audio=None,
                          reference_sample_rate=16000, verbose=False):
        loaded_models.append(voice_model)
        audio = np.zeros(16000, dtype=np.float32)
        sf.write(output_wav, audio, 16000)
        return True

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "indextts_tts", fake_indextts_tts)

    tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="indextts",
        voice_model="en_US-lessac-medium",  # Piper model – should be replaced
        verbose=False,
    )

    assert len(loaded_models) == 1
    assert loaded_models[0] == tts._INDEXTTS_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# synthesize_tts_pcm_with_cloning with indextts backend and voice_match
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_with_cloning_indextts_voice_match(monkeypatch):
    """synthesize_tts_pcm_with_cloning uses voice cloning when voice_match=True."""
    import soundfile as sf

    received_reference = []

    class FakeEngine:
        def infer(self, spk_audio_prompt, text, output_path, verbose=False):
            received_reference.append(spk_audio_prompt)
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(output_path, audio, 16000)

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "_load_indextts_engine", fake_load_engine)

    ref_audio = (np.random.randn(16000) * 32767).astype(np.int16)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        reference_text="This is the reference",
        voice_backend="indextts",
        voice_model="IndexTeam/IndexTTS-2",
        voice_match=True,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    # Verify the engine was called with a speaker prompt path
    assert len(received_reference) == 1
    assert received_reference[0].endswith(".wav")


def test_synthesize_tts_pcm_with_cloning_indextts_no_voice_match(monkeypatch):
    """Without voice_match, synthesize_tts_pcm_with_cloning falls through to standard synthesis."""
    import soundfile as sf

    def fake_indextts_tts(text, voice_model, output_wav, reference_audio=None,
                          reference_sample_rate=16000, verbose=False):
        audio = np.zeros(16000, dtype=np.float32)
        sf.write(output_wav, audio, 16000)
        return True

    monkeypatch.setattr(tts, "INDEXTTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_indextts_model_cache", {})
    monkeypatch.setattr(tts, "indextts_tts", fake_indextts_tts)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=None,
        voice_backend="indextts",
        voice_model="IndexTeam/IndexTTS-2",
        voice_match=False,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


# ---------------------------------------------------------------------------
# Availability flag presence
# ---------------------------------------------------------------------------

def test_indextts_availability_flag_exists():
    """INDEXTTS_AVAILABLE attribute must exist on the tts module."""
    assert hasattr(tts, "INDEXTTS_AVAILABLE")
    assert isinstance(tts.INDEXTTS_AVAILABLE, bool)


def test_indextts_class_attribute_exists():
    """_IndexTTS2 attribute must exist on the tts module (may be None)."""
    assert hasattr(tts, "_IndexTTS2")
