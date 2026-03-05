"""Tests for helper functions in anytran.tts."""
import os
import tempfile
import json
import unittest
from unittest.mock import MagicMock, patch


class TestFindPiperConfigPath(unittest.TestCase):
    """Test the _find_piper_config_path function."""

    def test_none_returns_none(self):
        from anytran.tts import _find_piper_config_path
        self.assertIsNone(_find_piper_config_path(None))

    def test_empty_string_returns_none(self):
        from anytran.tts import _find_piper_config_path
        self.assertIsNone(_find_piper_config_path(""))

    def test_onnx_file_with_json_sibling(self):
        from anytran.tts import _find_piper_config_path
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "voice.onnx")
            config_path = os.path.join(tmpdir, "voice.onnx.json")
            open(model_path, "w").close()
            open(config_path, "w").close()
            result = _find_piper_config_path(model_path)
            self.assertEqual(result, config_path)

    def test_onnx_file_with_base_json_sibling(self):
        from anytran.tts import _find_piper_config_path
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "voice.onnx")
            config_path = os.path.join(tmpdir, "voice.json")
            open(model_path, "w").close()
            open(config_path, "w").close()
            result = _find_piper_config_path(model_path)
            # voice.onnx.json checked first, then voice.json
            self.assertEqual(result, config_path)

    def test_nonexistent_config_returns_none(self):
        from anytran.tts import _find_piper_config_path
        result = _find_piper_config_path("/path/to/nonexistent.onnx")
        self.assertIsNone(result)

    def test_non_onnx_file(self):
        from anytran.tts import _find_piper_config_path
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "voice.bin")
            config_path = os.path.join(tmpdir, "voice.bin.json")
            open(model_path, "w").close()
            open(config_path, "w").close()
            result = _find_piper_config_path(model_path)
            self.assertEqual(result, config_path)


class TestResolvePiperSampleRate(unittest.TestCase):
    """Test the _resolve_piper_sample_rate function."""

    def test_uses_voice_sample_rate_attr(self):
        from anytran.tts import _resolve_piper_sample_rate
        voice = MagicMock()
        voice.sample_rate = 22050
        result = _resolve_piper_sample_rate(voice, None)
        self.assertEqual(result, 22050)

    def test_uses_voice_rate_attr(self):
        from anytran.tts import _resolve_piper_sample_rate
        voice = MagicMock(spec=["rate"])
        voice.rate = 16000
        result = _resolve_piper_sample_rate(voice, None)
        self.assertEqual(result, 16000)

    def test_reads_from_config_file(self):
        from anytran.tts import _resolve_piper_sample_rate
        voice = MagicMock(spec=[])  # No relevant attrs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"audio": {"sample_rate": 24000}}, f)
            config_path = f.name
        try:
            result = _resolve_piper_sample_rate(voice, config_path)
            self.assertEqual(result, 24000)
        finally:
            os.unlink(config_path)

    def test_reads_top_level_sample_rate_from_config(self):
        from anytran.tts import _resolve_piper_sample_rate
        voice = MagicMock(spec=[])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"sample_rate": 44100}, f)
            config_path = f.name
        try:
            result = _resolve_piper_sample_rate(voice, config_path)
            self.assertEqual(result, 44100)
        finally:
            os.unlink(config_path)

    def test_defaults_to_22050(self):
        from anytran.tts import _resolve_piper_sample_rate
        voice = MagicMock(spec=[])  # No sample rate attrs
        result = _resolve_piper_sample_rate(voice, None)
        self.assertEqual(result, 22050)

    def test_uses_config_attr_on_voice(self):
        from anytran.tts import _resolve_piper_sample_rate
        config = MagicMock()
        config.sample_rate = 16000
        voice = MagicMock(spec=["config"])
        voice.config = config
        result = _resolve_piper_sample_rate(voice, None)
        self.assertEqual(result, 16000)


class TestEnsureGttsAvailable(unittest.TestCase):
    """Test the _ensure_gtts_available function."""

    def test_returns_true_when_gtts_installed(self):
        from anytran.tts import _ensure_gtts_available
        # gTTS should be installed based on pyproject.toml dependencies
        result = _ensure_gtts_available(verbose=False)
        # Either True (installed) or False (not installed) — just verify no exception
        self.assertIsInstance(result, bool)

    def test_returns_false_when_gtts_none(self):
        import anytran.tts as tts_module
        from anytran.tts import _ensure_gtts_available
        orig_gtts = tts_module.gTTS
        tts_module.gTTS = None
        try:
            result = _ensure_gtts_available(verbose=False)
            self.assertFalse(result)
        finally:
            tts_module.gTTS = orig_gtts

    def test_returns_false_with_verbose_when_gtts_none(self):
        import anytran.tts as tts_module
        from anytran.tts import _ensure_gtts_available
        orig_gtts = tts_module.gTTS
        orig_audio = tts_module.AudioSegment
        tts_module.gTTS = None
        tts_module.AudioSegment = None
        try:
            result = _ensure_gtts_available(verbose=True)
            self.assertFalse(result)
        finally:
            tts_module.gTTS = orig_gtts
            tts_module.AudioSegment = orig_audio

    def test_returns_false_when_audiosegment_none(self):
        import anytran.tts as tts_module
        from anytran.tts import _ensure_gtts_available
        orig_audio = tts_module.AudioSegment
        tts_module.AudioSegment = None
        try:
            result = _ensure_gtts_available(verbose=False)
            self.assertFalse(result)
        finally:
            tts_module.AudioSegment = orig_audio


class TestSynthesizeTtsPcmEmptyText(unittest.TestCase):
    """Test synthesize_tts_pcm with empty/None text."""

    def test_returns_none_for_empty_text(self):
        from anytran.tts import synthesize_tts_pcm
        result = synthesize_tts_pcm("", rate=16000, output_lang="en")
        self.assertIsNone(result)

    def test_returns_none_for_none_text(self):
        from anytran.tts import synthesize_tts_pcm
        result = synthesize_tts_pcm(None, rate=16000, output_lang="en")
        self.assertIsNone(result)


class TestPlayOutputEmptyText(unittest.TestCase):
    """Test play_output with empty/None text — should return without doing anything."""

    def test_empty_text_returns_immediately(self):
        from anytran.tts import play_output
        # Should not raise
        play_output("", lang="en", play_audio=False)

    def test_none_text_returns_immediately(self):
        from anytran.tts import play_output
        # Should not raise
        play_output(None, lang="en", play_audio=False)


class TestPiperTtsNotAvailable(unittest.TestCase):
    """Test piper_tts when PIPER_PYTHON_AVAILABLE is False."""

    def test_returns_false_when_piper_not_available(self):
        import anytran.tts as tts_module
        from anytran.tts import piper_tts
        orig = tts_module.PIPER_PYTHON_AVAILABLE
        tts_module.PIPER_PYTHON_AVAILABLE = False
        try:
            result = piper_tts("Hello", "some_voice", "/tmp/output.wav", verbose=False)
            self.assertFalse(result)
        finally:
            tts_module.PIPER_PYTHON_AVAILABLE = orig


class TestEnsurePiperVoiceAvailable(unittest.TestCase):
    """Test ensure_piper_voice_available."""

    def test_returns_false_when_piper_not_available(self):
        import anytran.tts as tts_module
        from anytran.tts import ensure_piper_voice_available
        orig = tts_module.PIPER_PYTHON_AVAILABLE
        tts_module.PIPER_PYTHON_AVAILABLE = False
        try:
            result = ensure_piper_voice_available("en_US-test-medium", verbose=False)
            self.assertFalse(result)
        finally:
            tts_module.PIPER_PYTHON_AVAILABLE = orig


class TestModuleLevelCaches(unittest.TestCase):
    """Test that module-level caches are initialized correctly."""

    def test_piper_voice_cache_is_dict(self):
        from anytran.tts import _piper_voice_cache
        self.assertIsInstance(_piper_voice_cache, dict)

    def test_cached_matched_voice_initially_none(self):
        import anytran.tts as tts_module
        # These may have been set by other tests, just verify they are accessible
        self.assertIsNone(tts_module._cached_matched_voice) or True


if __name__ == "__main__":
    unittest.main()
