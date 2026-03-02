"""Tests for language-aware Piper voice auto-selection in synthesize_tts_pcm_with_cloning."""
import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def _setup_stubs():
    """Stub all heavy external packages needed to import anytran.tts."""
    stubs = [
        "soundfile", "librosa", "piper", "piper.voice",
        "gtts", "playsound3", "pydub",
    ]
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    piper_voice_mod = sys.modules["piper.voice"]
    if not hasattr(piper_voice_mod, "PiperVoice"):
        piper_voice_mod.PiperVoice = MagicMock()
    sys.modules["piper"].voice = piper_voice_mod

    gtts_mod = sys.modules["gtts"]
    if not hasattr(gtts_mod, "gTTS"):
        gtts_mod.gTTS = MagicMock()

    pydub_mod = sys.modules["pydub"]
    if not hasattr(pydub_mod, "AudioSegment"):
        pydub_mod.AudioSegment = MagicMock()

    playsound3_mod = sys.modules.get("playsound3")
    if playsound3_mod is None:
        playsound3_mod = types.ModuleType("playsound3")
        sys.modules["playsound3"] = playsound3_mod
    if not hasattr(playsound3_mod, "playsound"):
        playsound3_mod.playsound = MagicMock()


_setup_stubs()

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Remove any stubs that may have been registered by other test modules before us,
# then import the real anytran.tts so we can test its actual implementation.
for _mod_name in ("anytran.tts", "anytran.voice_matcher"):
    _existing = sys.modules.get(_mod_name)
    if _existing is not None and not hasattr(_existing, "__file__"):
        del sys.modules[_mod_name]

import anytran.tts as _tts_module  # noqa: E402  (must come after stubs)


def test_synthesize_tts_pcm_with_cloning_auto_selects_language_voice(tmp_path, monkeypatch):
    """
    When --voice-backend piper is used with a non-English output language and no
    explicit --voice-model, synthesize_tts_pcm_with_cloning should automatically
    select a voice appropriate for the output language instead of using the default
    English voice.
    """
    # Reset module-level state
    monkeypatch.setattr(_tts_module, "_cached_matched_voice", {})

    # Build a minimal voice table with both English and French voices
    table = [
        {"onnx_file": "en_US-lessac-medium.onnx", "pitch": 120, "gender": "male"},
        {"onnx_file": "fr_FR-tom-medium.onnx", "pitch": 115, "gender": "male"},
    ]
    table_path = tmp_path / "voice_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")

    # Load voice_matcher with our temp voice table
    import anytran.voice_matcher as voice_matcher
    monkeypatch.setattr(voice_matcher, "VOICE_TABLE_JSON_PATH", table_path)
    voice_matcher._load_piper_voices.cache_clear()

    # Patch the tts module to use our voice_matcher and a no-op synthesize_tts_pcm
    monkeypatch.setattr(_tts_module, "select_best_piper_voice", voice_matcher.select_best_piper_voice)

    used_voice = {}

    def fake_synthesize_tts_pcm(text, rate, output_lang, voice_backend="gtts", voice_model=None, verbose=False):
        used_voice["voice_model"] = voice_model
        used_voice["voice_backend"] = voice_backend
        return None  # Return value not important for this test

    monkeypatch.setattr(_tts_module, "synthesize_tts_pcm", fake_synthesize_tts_pcm)

    # Call with piper backend, default (English) voice model, French output language
    _tts_module.synthesize_tts_pcm_with_cloning(
        translated_text="Une tenue appropriée",
        rate=16000,
        output_lang="fr",
        voice_backend="piper",
        voice_model="en_US-lessac-medium",  # default / not explicitly provided by user
        voice_match=False,
        verbose=False,
    )

    # Should have auto-selected the French voice, NOT the English default
    assert used_voice.get("voice_backend") == "piper"
    assert used_voice.get("voice_model") == "fr_FR-tom-medium", (
        f"Expected French voice 'fr_FR-tom-medium' but got '{used_voice.get('voice_model')}'. "
        "The language-aware voice selection should have overridden the English default."
    )


def test_voice_match_runs_per_language(monkeypatch):
    """When --voice-match is enabled, matching should occur for each target language."""
    monkeypatch.setattr(_tts_module, "_cached_matched_voice", {})
    monkeypatch.setattr(_tts_module, "_cached_voice_features", None)

    calls = []

    def fake_extract(audio, sample_rate=16000, verbose=False):
        calls.append(("extract", sample_rate))
        return {"mean_pitch": 120.0, "gender": "male", "pitch_std": 0.0, "zcr": 0.1, "brightness": 2000.0}

    def fake_select(features, language, verbose=False):
        calls.append(("select", language))
        return f"{language}_voice"

    used_voice = []

    def fake_synthesize_tts_pcm(text, rate, output_lang, voice_backend="gtts", voice_model=None, verbose=False):
        used_voice.append((output_lang, voice_model))
        return None

    monkeypatch.setattr(_tts_module, "extract_voice_features", fake_extract)
    monkeypatch.setattr(_tts_module, "select_best_piper_voice", fake_select)
    monkeypatch.setattr(_tts_module, "synthesize_tts_pcm", fake_synthesize_tts_pcm)

    dummy_audio = np.zeros(16000, dtype=np.float32)

    # First: English output
    _tts_module.synthesize_tts_pcm_with_cloning(
        translated_text="hello",
        rate=16000,
        output_lang="en",
        reference_audio=dummy_audio,
        reference_sample_rate=16000,
        voice_backend="piper",
        voice_model="en_US-lessac-medium",
        voice_match=True,
        verbose=False,
    )

    # Second: French output in same process; should match separately
    _tts_module.synthesize_tts_pcm_with_cloning(
        translated_text="bonjour",
        rate=16000,
        output_lang="fr",
        reference_audio=dummy_audio,
        reference_sample_rate=16000,
        voice_backend="piper",
        voice_model="en_US-lessac-medium",
        voice_match=True,
        verbose=False,
    )

    # Voice selection should be run for both languages using cached features
    assert ("extract", 16000) in calls
    assert ("select", "en") in calls
    assert ("select", "fr") in calls
    # Each language should have its own matched voice
    assert ("en", "en_voice") in used_voice
    assert ("fr", "fr_voice") in used_voice


def test_synthesize_tts_pcm_with_cloning_explicit_voice_not_overridden(tmp_path, monkeypatch):
    """
    When the user explicitly provides a --voice-model that is not the default,
    it should not be overridden by the language-aware auto-selection.
    """
    monkeypatch.setattr(_tts_module, "_cached_matched_voice", {})

    used_voice = {}

    def fake_synthesize_tts_pcm(text, rate, output_lang, voice_backend="gtts", voice_model=None, verbose=False):
        used_voice["voice_model"] = voice_model
        return None

    monkeypatch.setattr(_tts_module, "synthesize_tts_pcm", fake_synthesize_tts_pcm)

    # Explicitly provided a non-default voice
    _tts_module.synthesize_tts_pcm_with_cloning(
        translated_text="Une tenue appropriée",
        rate=16000,
        output_lang="fr",
        voice_backend="piper",
        voice_model="en_US-ryan-high",  # explicit non-default voice
        voice_match=False,
        verbose=False,
    )

    assert used_voice.get("voice_model") == "en_US-ryan-high", (
        "An explicitly provided voice model should not be overridden by auto-selection."
    )


def test_synthesize_tts_pcm_with_cloning_english_output_keeps_default_voice(tmp_path, monkeypatch):
    """
    When output_lang is English, the default voice should not be auto-replaced.
    """
    monkeypatch.setattr(_tts_module, "_cached_matched_voice", {})

    used_voice = {}

    def fake_synthesize_tts_pcm(text, rate, output_lang, voice_backend="gtts", voice_model=None, verbose=False):
        used_voice["voice_model"] = voice_model
        return None

    monkeypatch.setattr(_tts_module, "synthesize_tts_pcm", fake_synthesize_tts_pcm)

    _tts_module.synthesize_tts_pcm_with_cloning(
        translated_text="Well deserved.",
        rate=16000,
        output_lang="en",
        voice_backend="piper",
        voice_model="en_US-lessac-medium",
        voice_match=False,
        verbose=False,
    )

    assert used_voice.get("voice_model") == "en_US-lessac-medium", (
        "English output should keep the default English voice."
    )


def test_synthesize_tts_pcm_auto_selects_language_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(_tts_module, "_cached_matched_voice", {})

    table = [
        {"onnx_file": "en_US-lessac-medium.onnx", "pitch": 120, "gender": "male"},
        {"onnx_file": "fr_FR-tom-medium.onnx", "pitch": 115, "gender": "male"},
    ]
    table_path = tmp_path / "voice_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")

    import anytran.voice_matcher as voice_matcher

    monkeypatch.setattr(voice_matcher, "VOICE_TABLE_JSON_PATH", table_path)
    voice_matcher._load_piper_voices.cache_clear()
    monkeypatch.setattr(_tts_module, "select_best_piper_voice", voice_matcher.select_best_piper_voice)

    used_voice = {}

    def fake_piper_tts(_text, voice_model, _output_wav, verbose=False):
        used_voice["voice_model"] = voice_model
        return True

    def fake_sf_read(_path):
        audio = np.ones(1600, dtype=np.float32) * 0.1
        return audio, 16000

    monkeypatch.setattr(_tts_module, "piper_tts", fake_piper_tts)
    monkeypatch.setattr(_tts_module.sf, "read", fake_sf_read, raising=False)

    pcm = _tts_module.synthesize_tts_pcm(
        translated_text="Une tenue appropriée",
        rate=16000,
        output_lang="fr",
        voice_backend="piper",
        voice_model="en_US-lessac-medium",
        verbose=False,
    )

    assert used_voice.get("voice_model") == "fr_FR-tom-medium", (
        "synthesize_tts_pcm should auto-select a French Piper voice when none is explicitly provided."
    )
    assert pcm is not None, "Piper synthesis should still return audio data after auto-selection."


def test_synthesize_tts_pcm_respects_explicit_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(_tts_module, "_cached_matched_voice", None)

    table_path = tmp_path / "voice_table.json"
    table_path.write_text(json.dumps([{"onnx_file": "fr_FR-tom-medium.onnx", "pitch": 115, "gender": "male"}]), encoding="utf-8")

    import anytran.voice_matcher as voice_matcher

    monkeypatch.setattr(voice_matcher, "VOICE_TABLE_JSON_PATH", table_path)
    voice_matcher._load_piper_voices.cache_clear()
    monkeypatch.setattr(_tts_module, "select_best_piper_voice", voice_matcher.select_best_piper_voice)

    used_voice = {}

    def fake_piper_tts(_text, voice_model, _output_wav, verbose=False):
        used_voice["voice_model"] = voice_model
        return True

    def fake_sf_read(_path):
        audio = np.ones(1600, dtype=np.float32) * 0.1
        return audio, 16000

    monkeypatch.setattr(_tts_module, "piper_tts", fake_piper_tts)
    monkeypatch.setattr(_tts_module.sf, "read", fake_sf_read, raising=False)

    pcm = _tts_module.synthesize_tts_pcm(
        translated_text="Tenue",
        rate=16000,
        output_lang="fr",
        voice_backend="piper",
        voice_model="fr_FR-tom-medium",
        verbose=False,
    )

    assert used_voice.get("voice_model") == "fr_FR-tom-medium", (
        "Explicit Piper voices must not be overridden by language-aware auto-selection."
    )
    assert pcm is not None
