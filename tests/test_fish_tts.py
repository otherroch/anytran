import sys
from pathlib import Path
import numpy as np

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anytran import tts


# ---------------------------------------------------------------------------
# _normalize_fish_model_name
# ---------------------------------------------------------------------------

def test_normalize_fish_model_name_default():
    """None or empty string should return the default s1-mini model."""
    assert tts._normalize_fish_model_name(None) == "fishaudio/openaudio-s1-mini"
    assert tts._normalize_fish_model_name("") == "fishaudio/openaudio-s1-mini"


def test_normalize_fish_model_name_alias():
    """fishaudio/s1-mini is an alias for fishaudio/openaudio-s1-mini."""
    assert tts._normalize_fish_model_name("fishaudio/s1-mini") == "fishaudio/openaudio-s1-mini"


def test_normalize_fish_model_name_canonical_passthrough():
    """Canonical names should be returned unchanged."""
    assert tts._normalize_fish_model_name("fishaudio/openaudio-s1-mini") == "fishaudio/openaudio-s1-mini"
    assert tts._normalize_fish_model_name("fishaudio/fish-speech-1.5") == "fishaudio/fish-speech-1.5"


# ---------------------------------------------------------------------------
# fish_tts – unavailable package
# ---------------------------------------------------------------------------

def test_fish_tts_not_available_returns_false(tmp_path, monkeypatch):
    """fish_tts returns False when fish-speech package is not installed."""
    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", False)

    result = tts.fish_tts(
        "Hello world",
        "fishaudio/openaudio-s1-mini",
        str(tmp_path / "output.wav"),
        verbose=False,
    )

    assert result is False


# ---------------------------------------------------------------------------
# Helpers for building mock fish-speech engine
# ---------------------------------------------------------------------------

def _make_mock_engine(sample_rate=44100, audio_length=44100):
    """Return a mock TTSInferenceEngine that yields a 1-second silence clip."""

    class MockInferenceResult:
        def __init__(self, code, audio, error=None):
            self.code = code
            self.audio = audio
            self.error = error

    class MockEngine:
        def __init__(self, sr=sample_rate, length=audio_length):
            self._sr = sr
            self._length = length

        def inference(self, request):
            audio = np.zeros(self._length, dtype=np.float32)
            yield MockInferenceResult(code="final", audio=(self._sr, audio))

    return MockEngine()


def _make_mock_fish_schema():
    """Return lightweight stand-ins for ServeReferenceAudio and ServeTTSRequest."""

    class FakeReferenceAudio:
        def __init__(self, audio, text):
            self.audio = audio
            self.text = text

    class FakeTTSRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    return FakeReferenceAudio, FakeTTSRequest


# ---------------------------------------------------------------------------
# fish_tts – successful synthesis (no voice cloning)
# ---------------------------------------------------------------------------

def test_fish_tts_basic_synthesis(tmp_path, monkeypatch):
    """fish_tts saves a WAV file and returns True on success."""
    output_wav = tmp_path / "output.wav"
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)

    engine = _make_mock_engine()
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: engine)

    result = tts.fish_tts(
        "Hello world",
        "fishaudio/openaudio-s1-mini",
        str(output_wav),
        verbose=False,
    )

    assert result is True
    assert output_wav.exists()


def test_fish_tts_uses_model_cache(tmp_path, monkeypatch):
    """fish_tts caches the engine so _load_fish_engine is only called once."""
    load_calls = {"count": 0}
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        load_calls["count"] += 1
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    model = "fishaudio/openaudio-s1-mini"
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"

    tts.fish_tts("Hello", model, str(out1), verbose=False)
    tts.fish_tts("World", model, str(out2), verbose=False)

    assert load_calls["count"] == 1  # loaded only once


def test_fish_tts_alias_resolved_before_cache_lookup(tmp_path, monkeypatch):
    """The s1-mini alias is resolved before the cache is consulted."""
    loaded_names = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        loaded_names.append(model_name)
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    out = tmp_path / "out.wav"
    tts.fish_tts("Hello", "fishaudio/s1-mini", str(out), verbose=False)

    assert loaded_names == ["fishaudio/openaudio-s1-mini"]


# ---------------------------------------------------------------------------
# fish_tts – voice cloning (with reference audio)
# ---------------------------------------------------------------------------

def test_fish_tts_voice_cloning(tmp_path, monkeypatch):
    """fish_tts passes reference audio to the engine when provided."""
    ref_received = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    class MockEngineCapturingRef:
        def inference(self, request):
            ref_received.extend(request.references)
            audio = np.zeros(44100, dtype=np.float32)

            class R:
                code = "final"
                audio = (44100, np.zeros(44100, dtype=np.float32))
                error = None

            yield R()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: MockEngineCapturingRef())

    ref_audio = np.clip(np.random.randn(16000), -1.0, 1.0).astype(np.float32)  # float32 in [-1.0, 1.0]
    out = tmp_path / "out.wav"

    result = tts.fish_tts(
        "Clone this voice",
        "fishaudio/openaudio-s1-mini",
        str(out),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        reference_text="This is the reference",
        verbose=False,
    )

    assert result is True
    assert len(ref_received) == 1
    assert ref_received[0].text == "This is the reference"


def test_fish_tts_int16_reference_audio(tmp_path, monkeypatch):
    """fish_tts normalises int16 reference audio to float32 before encoding."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())

    # int16 PCM reference audio (as used by the rest of anytran)
    ref_audio = (np.clip(np.random.randn(16000), -1.0, 1.0) * 32767).astype(np.int16)
    out = tmp_path / "out.wav"

    result = tts.fish_tts(
        "Hello",
        "fishaudio/openaudio-s1-mini",
        str(out),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        verbose=False,
    )

    assert result is True


# ---------------------------------------------------------------------------
# fish_tts – fish-speech-1.5 model
# ---------------------------------------------------------------------------

def test_fish_tts_fish_speech_15(tmp_path, monkeypatch):
    """fish_tts works with the fishaudio/fish-speech-1.5 model."""
    loaded_names = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        loaded_names.append(model_name)
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    out = tmp_path / "out.wav"
    result = tts.fish_tts("Hello", "fishaudio/fish-speech-1.5", str(out), verbose=False)

    assert result is True
    assert loaded_names == ["fishaudio/fish-speech-1.5"]


# ---------------------------------------------------------------------------
# fish_tts – engine load failure
# ---------------------------------------------------------------------------

def test_fish_tts_engine_load_failure_returns_false(tmp_path, monkeypatch):
    """fish_tts returns False when the engine cannot be loaded."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: None)

    result = tts.fish_tts(
        "Hello",
        "fishaudio/openaudio-s1-mini",
        str(tmp_path / "output.wav"),
        verbose=False,
    )

    assert result is False


# ---------------------------------------------------------------------------
# synthesize_tts_pcm – fish backend
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_fish_backend(monkeypatch):
    """synthesize_tts_pcm returns an int16 numpy array when using fish backend."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())

    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


def test_synthesize_tts_pcm_fish_default_model(monkeypatch):
    """synthesize_tts_pcm uses the default fish model when given a piper-style name."""
    loaded_names = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        loaded_names.append(model_name)
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="fish",
        voice_model="en_US-lessac-medium",  # piper-style name – should be replaced
        verbose=False,
    )

    assert result is not None
    assert loaded_names[0] == "fishaudio/openaudio-s1-mini"


# ---------------------------------------------------------------------------
# synthesize_tts_pcm_with_cloning – fish backend with voice_match
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_with_cloning_fish_voice_match(monkeypatch):
    """synthesize_tts_pcm_with_cloning does zero-shot cloning for fish + voice_match."""
    ref_texts_received = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    class MockEngineCapturingRef:
        def inference(self, request):
            for ref in request.references:
                ref_texts_received.append(ref.text)
            audio = np.zeros(44100, dtype=np.float32)

            class R:
                code = "final"
                audio = (44100, np.zeros(44100, dtype=np.float32))
                error = None

            yield R()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine",
                        lambda model_name, verbose=False: MockEngineCapturingRef())

    ref_audio = (np.clip(np.random.randn(16000), -1.0, 1.0) * 32767).astype(np.int16)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Synthesize this",
        rate=16000,
        output_lang="en",
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        reference_text="Reference text for cloning",
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        voice_match=True,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    assert len(ref_texts_received) == 1
    assert ref_texts_received[0] == "Reference text for cloning"


def test_synthesize_tts_pcm_with_cloning_fish_no_voice_match(monkeypatch):
    """With fish backend but voice_match=False, basic synthesis is used (no cloning)."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())
    # Reset piper voice cache so we do not accidentally pick up a cached piper voice
    monkeypatch.setattr(tts, "_cached_matched_voice", None)
    monkeypatch.setattr(tts, "_cached_output_lang", None)

    ref_audio = (np.clip(np.random.randn(16000), -1.0, 1.0) * 32767).astype(np.int16)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        voice_match=False,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


def test_synthesize_tts_pcm_with_cloning_fish_voice_match_no_ref(monkeypatch):
    """With voice_match but no reference audio, basic synthesis is used."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())
    monkeypatch.setattr(tts, "_cached_matched_voice", None)
    monkeypatch.setattr(tts, "_cached_output_lang", None)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=None,  # no reference audio
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        voice_match=True,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
