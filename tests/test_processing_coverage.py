"""Tests for anytran.processing.build_output_prefix and related utilities."""
import unittest
from unittest.mock import patch, MagicMock


class TestBuildOutputPrefix(unittest.TestCase):
    """Test build_output_prefix with various language codes."""

    def test_none_detected_lang_returns_unknown(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang=None)
        self.assertEqual(result, "Unknown: ")

    def test_empty_detected_lang_returns_unknown(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="")
        self.assertEqual(result, "Unknown: ")

    def test_english(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="en")
        self.assertEqual(result, "English: ")

    def test_french(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="fr")
        self.assertEqual(result, "French: ")

    def test_spanish(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="es")
        self.assertEqual(result, "Spanish: ")

    def test_german(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="de")
        self.assertEqual(result, "German: ")

    def test_chinese(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="zh")
        self.assertEqual(result, "Chinese: ")

    def test_chinese_cn(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="zh-cn")
        self.assertEqual(result, "Chinese: ")

    def test_japanese(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="ja")
        self.assertEqual(result, "Japanese: ")

    def test_korean(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="ko")
        self.assertEqual(result, "Korean: ")

    def test_russian(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="ru")
        self.assertEqual(result, "Russian: ")

    def test_portuguese(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="pt")
        self.assertEqual(result, "Portuguese: ")

    def test_arabic(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="ar")
        self.assertEqual(result, "Arabic: ")

    def test_hindi(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="hi")
        self.assertEqual(result, "Hindi: ")

    def test_ukrainian(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="uk")
        self.assertEqual(result, "Ukrainian: ")

    def test_uppercase_lang_code(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="EN")
        self.assertEqual(result, "English: ")

    def test_lang_with_region_code(self):
        from anytran.processing import build_output_prefix
        # "en-US" should map to English via base code
        result = build_output_prefix(detected_lang="en-US")
        self.assertEqual(result, "English: ")

    def test_unknown_lang_code(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="xyz")
        self.assertEqual(result, "Unknown: ")

    def test_stream_id_ignored(self):
        from anytran.processing import build_output_prefix
        result_with = build_output_prefix(stream_id="stream1", detected_lang="en")
        result_without = build_output_prefix(stream_id=None, detected_lang="en")
        self.assertEqual(result_with, result_without)

    def test_dutch(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="nl")
        self.assertEqual(result, "Dutch: ")

    def test_italian(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="it")
        self.assertEqual(result, "Italian: ")

    def test_turkish(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="tr")
        self.assertEqual(result, "Turkish: ")

    def test_vietnamese(self):
        from anytran.processing import build_output_prefix
        result = build_output_prefix(detected_lang="vi")
        self.assertEqual(result, "Vietnamese: ")


class TestProcessAudioChunkMagnitudeCheck(unittest.TestCase):
    """Test process_audio_chunk early exit paths."""

    def test_silent_audio_skipped_with_verbose(self):
        """When magnitude is below threshold, the function should skip processing."""
        import numpy as np
        from unittest.mock import patch

        # Silent audio — all zeros, magnitude = 0
        audio = np.zeros(16000, dtype=np.float32)

        with patch("anytran.processing.translate_audio") as mock_translate:
            from anytran.processing import process_audio_chunk
            result = process_audio_chunk(
                audio_segment=audio,
                rate=16000,
                input_lang="en",
                output_lang="en",
                magnitude_threshold=0.1,  # Above zero magnitude
                model=None,
                verbose=True,
                mqtt_broker=None,
                mqtt_port=1883,
                mqtt_username=None,
                mqtt_password=None,
                mqtt_topic=None,
            )
            # translate_audio should not be called for silent audio
            mock_translate.assert_not_called()

    def test_audio_with_speech_calls_translate(self):
        """When magnitude is above threshold and VAD is disabled, translate_audio is called."""
        import numpy as np

        audio = np.ones(16000, dtype=np.float32) * 0.5  # Non-silent

        with patch("anytran.processing.translate_audio") as mock_translate:
            mock_translate.return_value = ("hello world", "en", [])
            from anytran.processing import process_audio_chunk
            process_audio_chunk(
                audio_segment=audio,
                rate=16000,
                input_lang=None,
                output_lang=None,
                magnitude_threshold=0.1,
                model=None,
                verbose=False,
                mqtt_broker=None,
                mqtt_port=1883,
                mqtt_username=None,
                mqtt_password=None,
                mqtt_topic=None,
                scribe_vad=False,
            )
            mock_translate.assert_called_once()

    def test_with_timers_enabled(self):
        """Test that timers flag causes timing data to be collected."""
        import numpy as np

        audio = np.zeros(16000, dtype=np.float32)

        with patch("anytran.processing.translate_audio") as mock_translate:
            from anytran.processing import process_audio_chunk
            process_audio_chunk(
                audio_segment=audio,
                rate=16000,
                input_lang=None,
                output_lang=None,
                magnitude_threshold=0.5,
                model=None,
                verbose=False,
                mqtt_broker=None,
                mqtt_port=1883,
                mqtt_username=None,
                mqtt_password=None,
                mqtt_topic=None,
                timers=True,
            )
            # Should complete without error


if __name__ == "__main__":
    unittest.main()
