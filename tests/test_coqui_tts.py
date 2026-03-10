import sys
from pathlib import Path
import numpy as np

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anytran import tts


# ---------------------------------------------------------------------------
# Constants / availability flags
# ---------------------------------------------------------------------------

def test_coqui_default_model_constant():
    """_COQUI_DEFAULT_MODEL should point to the XTTS v2 repo."""
    assert tts._COQUI_DEFAULT_MODEL == "tts_models/multilingual/multi-dataset/xtts_v2"


def test_coqui_availability_flag_exists():
    """COQUI_TTS_AVAILABLE attribute must exist on the tts module."""
    assert hasattr(tts, "COQUI_TTS_AVAILABLE")
    assert isinstance(tts.COQUI_TTS_AVAILABLE, bool)


def test_coqui_class_attribute_exists():
    """_CoquiTTS attribute must exist on the tts module (may be None)."""
    assert hasattr(tts, "_CoquiTTS")


# ---------------------------------------------------------------------------
# _map_to_coqui_language
# ---------------------------------------------------------------------------

def test_map_to_coqui_language_english():
    assert tts._map_to_coqui_language("en") == "en"
    assert tts._map_to_coqui_language("en-US") == "en"
    assert tts._map_to_coqui_language("en-GB") == "en"


def test_map_to_coqui_language_none_defaults_to_english():
    assert tts._map_to_coqui_language(None) == "en"
    assert tts._map_to_coqui_language("") == "en"


def test_map_to_coqui_language_chinese_mapped():
    assert tts._map_to_coqui_language("zh") == "zh-cn"
    assert tts._map_to_coqui_language("zh-cn") == "zh-cn"
    assert tts._map_to_coqui_language("zh-tw") == "zh-cn"
    assert tts._map_to_coqui_language("zh-hk") == "zh-cn"
    assert tts._map_to_coqui_language("zho") == "zh-cn"
    assert tts._map_to_coqui_language("cmn") == "zh-cn"


def test_map_to_coqui_language_supported_codes():
    for code in ("fr", "de", "es", "it", "pt", "pl", "tr", "ru", "nl",
                 "cs", "ar", "hu", "ko", "ja", "hi"):
        assert tts._map_to_coqui_language(code) == code


def test_map_to_coqui_language_unsupported_falls_back_to_english():
    assert tts._map_to_coqui_language("xx") == "en"
    assert tts._map_to_coqui_language("xyz") == "en"


def test_map_to_coqui_language_bcp47_with_region():
    assert tts._map_to_coqui_language("fr-CA") == "fr"
    assert tts._map_to_coqui_language("de-DE") == "de"


# ---------------------------------------------------------------------------
# coqui_tts – unavailable package
# ---------------------------------------------------------------------------

def test_coqui_tts_not_available_returns_false(tmp_path, monkeypatch):
    """coqui_tts returns False when coqui-tts package is not installed."""
    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", False)

    result = tts.coqui_tts(
        "Hello world",
        tts._COQUI_DEFAULT_MODEL,
        "en",
        str(tmp_path / "output.wav"),
        verbose=False,
    )

    assert result is False


# ---------------------------------------------------------------------------
# coqui_tts – with a fake engine (no reference audio)
# ---------------------------------------------------------------------------

def test_coqui_tts_without_reference_audio(tmp_path, monkeypatch):
    """coqui_tts synthesizes without voice cloning when reference_audio is None."""
    import soundfile as sf

    output_wav = tmp_path / "output.wav"

    class FakeEngine:
        is_multi_lingual = True

        def tts_to_file(self, text, file_path, language=None, speaker_wav=None):
            audio = np.zeros(22050, dtype=np.float32)
            sf.write(file_path, audio, 22050)

        def to(self, device):
            return self

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "_load_coqui_engine", fake_load_engine)

    result = tts.coqui_tts(
        "Hello world",
        tts._COQUI_DEFAULT_MODEL,
        "en",
        str(output_wav),
        reference_audio=None,
        verbose=False,
    )

    assert result is True
    assert output_wav.exists()


def test_coqui_tts_without_reference_audio_multi_speaker_uses_default(tmp_path, monkeypatch):
    """coqui_tts passes the first available speaker when the engine is multi-speaker
    and no reference_audio is provided (fixes 'Neither speaker_wav nor speaker_id
    was specified' error for XTTS v2 without --voice-match)."""
    import soundfile as sf

    output_wav = tmp_path / "output.wav"
    received_kwargs = {}

    class FakeMultiSpeakerEngine:
        is_multi_lingual = True
        speakers = ["Claribel Dervla", "Daisy Studious", "Gracie Wise"]

        def tts_to_file(self, text, file_path, language=None, speaker=None, **kwargs):
            received_kwargs["speaker"] = speaker
            received_kwargs["language"] = language
            audio = np.zeros(22050, dtype=np.float32)
            sf.write(file_path, audio, 22050)

        def to(self, device):
            return self

    def fake_load_engine(model_name, verbose=False):
        return FakeMultiSpeakerEngine()

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "_load_coqui_engine", fake_load_engine)

    result = tts.coqui_tts(
        "Bonjour le monde",
        tts._COQUI_DEFAULT_MODEL,
        "fr",
        str(output_wav),
        reference_audio=None,
        verbose=False,
    )

    assert result is True
    assert output_wav.exists()
    # The first built-in speaker must have been forwarded to tts_to_file
    assert received_kwargs.get("speaker") == "Claribel Dervla"
    assert received_kwargs.get("language") == "fr"


# ---------------------------------------------------------------------------
# coqui_tts – with reference audio (voice cloning)
# ---------------------------------------------------------------------------

def test_coqui_tts_with_reference_audio(tmp_path, monkeypatch):
    """coqui_tts performs voice cloning when reference_audio is provided."""
    import soundfile as sf

    output_wav = tmp_path / "output.wav"
    received_kwargs = {}

    class FakeEngine:
        is_multi_lingual = True

        def tts_to_file(self, text, file_path, language=None, speaker_wav=None):
            received_kwargs["speaker_wav"] = speaker_wav
            received_kwargs["language"] = language
            audio = np.zeros(22050, dtype=np.float32)
            sf.write(file_path, audio, 22050)

        def to(self, device):
            return self

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "_load_coqui_engine", fake_load_engine)

    ref_audio = (np.zeros(16000) * 32767).astype(np.int16)

    result = tts.coqui_tts(
        "Hello world",
        tts._COQUI_DEFAULT_MODEL,
        "en",
        str(output_wav),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        verbose=False,
    )

    assert result is True
    assert output_wav.exists()
    assert received_kwargs.get("speaker_wav") is not None
    assert received_kwargs.get("speaker_wav").endswith(".wav")
    assert received_kwargs.get("language") == "en"


def test_coqui_tts_reference_audio_float_range(tmp_path, monkeypatch):
    """coqui_tts normalises float32 reference audio into [-1, 1] before writing."""
    import soundfile as sf
    import os

    received_paths = []

    class FakeEngine:
        is_multi_lingual = True

        def tts_to_file(self, text, file_path, language=None, speaker_wav=None):
            received_paths.append(speaker_wav)
            audio = np.zeros(22050, dtype=np.float32)
            sf.write(file_path, audio, 22050)

        def to(self, device):
            return self

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "_load_coqui_engine", fake_load_engine)

    # float32 already in [-1, 1]
    ref_audio = np.random.uniform(-1.0, 1.0, 16000).astype(np.float32)

    result = tts.coqui_tts(
        "Test",
        tts._COQUI_DEFAULT_MODEL,
        "en",
        str(tmp_path / "out.wav"),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        verbose=False,
    )

    assert result is True
    assert len(received_paths) == 1
    # Temp reference file is cleaned up after the call
    assert not os.path.exists(received_paths[0])


# ---------------------------------------------------------------------------
# coqui_tts – caching
# ---------------------------------------------------------------------------

def test_coqui_tts_uses_cache(tmp_path, monkeypatch):
    """coqui_tts caches the engine so it is only loaded once."""
    import soundfile as sf

    load_count = {"count": 0}

    class FakeEngine:
        is_multi_lingual = True

        def tts_to_file(self, text, file_path, language=None, speaker_wav=None):
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(file_path, audio, 16000)

        def to(self, device):
            return self

    def fake_load_engine(model_name, verbose=False):
        load_count["count"] += 1
        return FakeEngine()

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "_load_coqui_engine", fake_load_engine)

    model_name = tts._COQUI_DEFAULT_MODEL

    tts.coqui_tts("Hello", model_name, "en", str(tmp_path / "out1.wav"), verbose=False)
    assert load_count["count"] == 1

    tts.coqui_tts("World", model_name, "en", str(tmp_path / "out2.wav"), verbose=False)
    assert load_count["count"] == 1


def test_coqui_tts_default_model_used_when_none(tmp_path, monkeypatch):
    """coqui_tts uses _COQUI_DEFAULT_MODEL when voice_model is None."""
    import soundfile as sf

    loaded_models = []

    class FakeEngine:
        is_multi_lingual = True

        def tts_to_file(self, text, file_path, language=None, speaker_wav=None):
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(file_path, audio, 16000)

        def to(self, device):
            return self

    def fake_load_engine(model_name, verbose=False):
        loaded_models.append(model_name)
        return FakeEngine()

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "_load_coqui_engine", fake_load_engine)

    tts.coqui_tts("Hello", None, "en", str(tmp_path / "out.wav"), verbose=False)

    assert loaded_models == [tts._COQUI_DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# synthesize_tts_pcm with coqui backend
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_with_coqui_backend(tmp_path, monkeypatch):
    """synthesize_tts_pcm with coqui backend returns int16 PCM array."""
    import soundfile as sf

    def fake_coqui_tts(text, voice_model, output_lang, output_wav,
                       reference_audio=None, reference_sample_rate=16000, verbose=False):
        audio = np.zeros(22050, dtype=np.float32)
        sf.write(output_wav, audio, 22050)
        return True

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "coqui_tts", fake_coqui_tts)

    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="coqui",
        voice_model=tts._COQUI_DEFAULT_MODEL,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


def test_synthesize_tts_pcm_coqui_replaces_non_coqui_model(monkeypatch):
    """synthesize_tts_pcm replaces non-coqui model names with the default."""
    import soundfile as sf

    loaded_models = []

    def fake_coqui_tts(text, voice_model, output_lang, output_wav,
                       reference_audio=None, reference_sample_rate=16000, verbose=False):
        loaded_models.append(voice_model)
        audio = np.zeros(16000, dtype=np.float32)
        sf.write(output_wav, audio, 16000)
        return True

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "coqui_tts", fake_coqui_tts)

    tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="coqui",
        voice_model="en_US-lessac-medium",  # Piper model – should be replaced
        verbose=False,
    )

    assert len(loaded_models) == 1
    assert loaded_models[0] == tts._COQUI_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# synthesize_tts_pcm_with_cloning with coqui backend and voice_match
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_with_cloning_coqui_voice_match(monkeypatch):
    """synthesize_tts_pcm_with_cloning uses voice cloning when voice_match=True."""
    import soundfile as sf

    received_reference = []

    class FakeEngine:
        is_multi_lingual = True

        def tts_to_file(self, text, file_path, language=None, speaker_wav=None):
            received_reference.append(speaker_wav)
            audio = np.zeros(22050, dtype=np.float32)
            sf.write(file_path, audio, 22050)

        def to(self, device):
            return self

    def fake_load_engine(model_name, verbose=False):
        return FakeEngine()

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "_load_coqui_engine", fake_load_engine)

    ref_audio = (np.random.randn(16000) * 32767).astype(np.int16)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        reference_text="This is the reference",
        voice_backend="coqui",
        voice_model=tts._COQUI_DEFAULT_MODEL,
        voice_match=True,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    assert len(received_reference) == 1
    assert received_reference[0].endswith(".wav")


def test_synthesize_tts_pcm_with_cloning_coqui_no_voice_match(monkeypatch):
    """Without voice_match, synthesize_tts_pcm_with_cloning falls through to standard synthesis."""
    import soundfile as sf

    def fake_coqui_tts(text, voice_model, output_lang, output_wav,
                       reference_audio=None, reference_sample_rate=16000, verbose=False):
        audio = np.zeros(22050, dtype=np.float32)
        sf.write(output_wav, audio, 22050)
        return True

    monkeypatch.setattr(tts, "COQUI_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_coqui_model_cache", {})
    monkeypatch.setattr(tts, "coqui_tts", fake_coqui_tts)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=None,
        voice_backend="coqui",
        voice_model=tts._COQUI_DEFAULT_MODEL,
        voice_match=False,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


# ---------------------------------------------------------------------------
# Language mapping constants
# ---------------------------------------------------------------------------

def test_coqui_xtts_languages_set():
    """_COQUI_XTTS_LANGUAGES should contain expected language codes."""
    expected = {"en", "fr", "de", "es", "it", "pt", "pl", "tr", "ru",
                "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"}
    assert expected == tts._COQUI_XTTS_LANGUAGES


def test_coqui_lang_map_chinese():
    """_COQUI_LANG_MAP must map all Chinese variants to zh-cn."""
    assert tts._COQUI_LANG_MAP["zh"] == "zh-cn"
    assert tts._COQUI_LANG_MAP["zh-tw"] == "zh-cn"
    assert tts._COQUI_LANG_MAP["zh-hk"] == "zh-cn"
    assert tts._COQUI_LANG_MAP["zho"] == "zh-cn"
    assert tts._COQUI_LANG_MAP["cmn"] == "zh-cn"
