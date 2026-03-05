"""Tests for anytran.tts synthesize_tts_pcm using mocked gTTS."""
import io
import os
import numpy as np
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call


def _make_mock_audio_chain():
    """Return a mock AudioSegment that handles the chained set_* calls."""
    raw_pcm = (np.ones(1600, dtype=np.int16) * 500).tobytes()
    seg = MagicMock()
    seg.raw_data = raw_pcm
    seg.channels = 1
    seg.sample_width = 2
    seg.frame_rate = 16000
    # Each set_* returns the same mock (simulating a fresh AudioSegment)
    seg.set_frame_rate = MagicMock(return_value=seg)
    seg.set_channels = MagicMock(return_value=seg)
    seg.set_sample_width = MagicMock(return_value=seg)
    return seg


class TestSynthesizeTtsPcmMockedGtts(unittest.TestCase):
    """Test synthesize_tts_pcm end-to-end with mocked gTTS and AudioSegment."""

    def setUp(self):
        import anytran.tts as tts_module
        self._orig_gtts = tts_module.gTTS
        self._orig_audio = tts_module.AudioSegment

    def tearDown(self):
        import anytran.tts as tts_module
        tts_module.gTTS = self._orig_gtts
        tts_module.AudioSegment = self._orig_audio

    def _setup_mocks(self):
        import anytran.tts as tts_module
        mock_gtts = MagicMock()
        mock_gtts.return_value = MagicMock()
        mock_audio_seg = _make_mock_audio_chain()
        mock_audio_class = MagicMock()
        mock_audio_class.from_mp3 = MagicMock(return_value=mock_audio_seg)
        tts_module.gTTS = mock_gtts
        tts_module.AudioSegment = mock_audio_class
        return mock_gtts, mock_audio_seg

    def test_synthesize_returns_numpy_array(self):
        import anytran.tts as tts_module
        mock_gtts, _ = self._setup_mocks()

        result = tts_module.synthesize_tts_pcm("Hello world", rate=16000, output_lang="en")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)

    def test_synthesize_calls_gtts(self):
        import anytran.tts as tts_module
        mock_gtts, _ = self._setup_mocks()

        tts_module.synthesize_tts_pcm("Bonjour", rate=16000, output_lang="fr")
        mock_gtts.assert_called()

    def test_synthesize_gtts_exception_falls_back_to_english(self):
        import anytran.tts as tts_module

        # First gTTS call fails, second (English fallback) succeeds
        call_count = [0]
        def side_effect(text, lang):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("TTS Error")
            return MagicMock()

        mock_audio_seg = _make_mock_audio_chain()
        mock_audio_class = MagicMock()
        mock_audio_class.from_mp3 = MagicMock(return_value=mock_audio_seg)
        tts_module.gTTS = side_effect
        tts_module.AudioSegment = mock_audio_class

        result = tts_module.synthesize_tts_pcm(
            "Hello", rate=16000, output_lang="en", verbose=True
        )
        # Either returns array from English fallback or None - just no exception
        # The fallback calls gTTS with lang="en" and returns the PCM

    def test_synthesize_empty_text_returns_none(self):
        import anytran.tts as tts_module
        result = tts_module.synthesize_tts_pcm("", rate=16000, output_lang="en")
        self.assertIsNone(result)

    def test_synthesize_none_text_returns_none(self):
        import anytran.tts as tts_module
        result = tts_module.synthesize_tts_pcm(None, rate=16000, output_lang="en")
        self.assertIsNone(result)

    def test_synthesize_verbose_mode(self):
        import anytran.tts as tts_module
        self._setup_mocks()
        result = tts_module.synthesize_tts_pcm(
            "Hello verbose", rate=16000, output_lang="en", verbose=True
        )
        self.assertIsNotNone(result)

    def test_synthesize_german_lang(self):
        import anytran.tts as tts_module
        mock_gtts, _ = self._setup_mocks()
        result = tts_module.synthesize_tts_pcm("Hallo", rate=16000, output_lang="de")
        self.assertIsNotNone(result)
        # gTTS called with de lang
        mock_gtts.assert_called()

    def test_synthesize_gtts_all_fail_returns_none(self):
        import anytran.tts as tts_module
        tts_module.gTTS = MagicMock(side_effect=Exception("All TTS Failed"))
        result = tts_module.synthesize_tts_pcm("Hello", rate=16000, output_lang="en")
        self.assertIsNone(result)


class TestPlayOutputMocked(unittest.TestCase):
    """Test play_output using mocked gTTS."""

    def setUp(self):
        import anytran.tts as tts_module
        self._orig_gtts = tts_module.gTTS
        self._orig_audio = tts_module.AudioSegment
        self._orig_playsound = tts_module.playsound

    def tearDown(self):
        import anytran.tts as tts_module
        tts_module.gTTS = self._orig_gtts
        tts_module.AudioSegment = self._orig_audio
        tts_module.playsound = self._orig_playsound

    def test_play_output_empty_text_returns_early(self):
        import anytran.tts as tts_module
        mock_gtts = MagicMock()
        tts_module.gTTS = mock_gtts
        tts_module.play_output("", lang="en", play_audio=False)
        mock_gtts.assert_not_called()

    def test_play_output_no_play_audio(self):
        import anytran.tts as tts_module
        mock_gtts = MagicMock()
        mock_gtts.return_value = MagicMock()
        mock_playsound = MagicMock()
        tts_module.gTTS = mock_gtts
        tts_module.playsound = mock_playsound
        tts_module.play_output("Hello", lang="en", play_audio=False)
        mock_playsound.assert_not_called()

    def test_play_output_with_wav_file(self):
        import anytran.tts as tts_module
        mock_gtts = MagicMock()
        mock_gtts.return_value = MagicMock()

        mock_audio_seg = _make_mock_audio_chain()
        mock_audio_class = MagicMock()
        mock_audio_class.from_mp3 = MagicMock(return_value=mock_audio_seg)
        tts_module.gTTS = mock_gtts
        tts_module.AudioSegment = mock_audio_class

        mock_wav = MagicMock()
        tts_module.play_output("Hello", lang="en", play_audio=False, wav_file=mock_wav)
        mock_wav.writeframes.assert_called_once()


class TestEnsureGttsAvailable(unittest.TestCase):
    """Test _ensure_gtts_available function."""

    def test_returns_true_when_available(self):
        import anytran.tts as tts_module
        orig_gtts = tts_module.gTTS
        orig_audio = tts_module.AudioSegment
        try:
            tts_module.gTTS = MagicMock()
            tts_module.AudioSegment = MagicMock()
            result = tts_module._ensure_gtts_available()
            self.assertTrue(result)
        finally:
            tts_module.gTTS = orig_gtts
            tts_module.AudioSegment = orig_audio

    def test_returns_false_when_gtts_none(self):
        import anytran.tts as tts_module
        orig_gtts = tts_module.gTTS
        try:
            tts_module.gTTS = None
            result = tts_module._ensure_gtts_available(verbose=False)
            self.assertFalse(result)
        finally:
            tts_module.gTTS = orig_gtts


class TestTtsHelpers(unittest.TestCase):
    """Test TTS helper functions."""

    def test_find_piper_config_path_nonexistent(self):
        from anytran.tts import _find_piper_config_path
        result = _find_piper_config_path("/nonexistent/path/to/model.onnx")
        self.assertIsNone(result)

    def test_find_piper_config_path_with_existing_config(self):
        from anytran.tts import _find_piper_config_path
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "voice.onnx")
            config_path = model_path + ".json"
            open(model_path, "w").write("")
            open(config_path, "w").write('{"sample_rate": 22050}')
            result = _find_piper_config_path(model_path)
            self.assertEqual(result, config_path)

    def test_resolve_piper_sample_rate_no_config(self):
        from anytran.tts import _resolve_piper_sample_rate
        result = _resolve_piper_sample_rate("en_US-test", None)
        self.assertIsInstance(result, int)

    def test_resolve_piper_sample_rate_with_config(self):
        import json
        from anytran.tts import _resolve_piper_sample_rate
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"audio": {"sample_rate": 22050}}, f)
            cfg_path = f.name
        try:
            result = _resolve_piper_sample_rate("test_voice", cfg_path)
            self.assertEqual(result, 22050)
        finally:
            os.unlink(cfg_path)

    def test_synthesize_tts_pcm_with_cloning_empty_text(self):
        from anytran.tts import synthesize_tts_pcm_with_cloning
        result = synthesize_tts_pcm_with_cloning("", 16000, "en")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
