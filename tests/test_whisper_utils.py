"""Tests for utility functions in anytran.whisper_backend."""
import unittest
from unittest.mock import MagicMock, patch

# test_looptran.py replaces a few whisper_backend attributes with MagicMock.
# Use the real function references saved by conftest.py before that happens.
from tests.conftest import _real_whisper_backend_funcs as _WBF

_derive_whispercpp_model_name = _WBF["_derive_whispercpp_model_name"]
_get_with_override = _WBF["_get_with_override"]
_extract_detected_language_from_output = _WBF["_extract_detected_language_from_output"]
_extract_detected_language_from_result = _WBF["_extract_detected_language_from_result"]
get_effective_backend = _WBF["get_effective_backend"]
is_hallucination = _WBF["is_hallucination"]


class TestIsHallucination(unittest.TestCase):
    """Test the is_hallucination function."""

    def test_none_is_hallucination(self):
        self.assertTrue(is_hallucination(None))

    def test_empty_string_is_hallucination(self):
        self.assertTrue(is_hallucination(""))

    def test_too_short_is_hallucination(self):
        self.assertTrue(is_hallucination("hi"))

    def test_known_phrase_is_hallucination(self):
        self.assertTrue(is_hallucination("Thank you for watching this video!"))

    def test_please_subscribe_is_hallucination(self):
        self.assertTrue(is_hallucination("Please subscribe to my channel"))

    def test_see_you_next_video_is_hallucination(self):
        self.assertTrue(is_hallucination("See you in the next video!"))

    def test_normal_speech_not_hallucination(self):
        self.assertFalse(is_hallucination("The weather is nice today in Paris"))

    def test_repeated_words_is_hallucination(self):
        # High repetition → unique ratio < 0.5
        self.assertTrue(is_hallucination("the the the the the the the the the"))

    def test_diverse_sentence_not_hallucination(self):
        text = "This is a completely normal and meaningful sentence without repetition"
        self.assertFalse(is_hallucination(text))


class TestGetWithOverride(unittest.TestCase):
    """Test the _get_with_override utility function."""

    def test_override_wins_over_dict(self):
        result = _get_with_override("explicit", {"key": "from_dict"}, "key")
        self.assertEqual(result, "explicit")

    def test_fallback_to_dict_when_no_override(self):
        result = _get_with_override(None, {"key": "from_dict"}, "key")
        self.assertEqual(result, "from_dict")

    def test_default_when_key_missing_and_no_override(self):
        result = _get_with_override(None, {}, "key", default="fallback_default")
        self.assertEqual(result, "fallback_default")

    def test_none_default_when_key_missing(self):
        result = _get_with_override(None, {}, "missing_key")
        self.assertIsNone(result)

    def test_zero_override_wins(self):
        # 0 is a valid override (not None)
        result = _get_with_override(0, {"key": 99}, "key")
        # 0 is not None, so override wins
        self.assertEqual(result, 0)


class TestDeriveWhisperCppModelName(unittest.TestCase):
    """Test the _derive_whispercpp_model_name function."""

    def test_none_returns_none(self):
        self.assertIsNone(_derive_whispercpp_model_name(None))

    def test_ggml_medium_bin(self):
        result = _derive_whispercpp_model_name("/models/ggml-medium.bin")
        self.assertEqual(result, "medium")

    def test_ggml_small_bin(self):
        result = _derive_whispercpp_model_name("ggml-small.bin")
        self.assertEqual(result, "small")

    def test_known_name_no_ggml_prefix(self):
        result = _derive_whispercpp_model_name("/path/to/medium")
        self.assertEqual(result, "medium")

    def test_unknown_name_returns_none(self):
        result = _derive_whispercpp_model_name("/path/to/custom_model")
        self.assertIsNone(result)

    def test_tiny_en(self):
        result = _derive_whispercpp_model_name("ggml-tiny.en.bin")
        self.assertEqual(result, "tiny.en")

    def test_large_v1(self):
        result = _derive_whispercpp_model_name("ggml-large-v1.bin")
        self.assertEqual(result, "large-v1")


class TestExtractDetectedLanguageFromOutput(unittest.TestCase):
    """Test the _extract_detected_language_from_output function."""

    def test_none_returns_none(self):
        self.assertIsNone(_extract_detected_language_from_output(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_extract_detected_language_from_output(""))

    def test_detects_language_from_output(self):
        output = "Some text... auto-detected language: fr, probability = 0.97"
        result = _extract_detected_language_from_output(output)
        self.assertEqual(result, "fr")

    def test_case_insensitive(self):
        output = "Auto-Detected Language: EN"
        result = _extract_detected_language_from_output(output)
        self.assertEqual(result, "en")

    def test_no_language_in_output_returns_none(self):
        result = _extract_detected_language_from_output("normal output with no language detection")
        self.assertIsNone(result)


class TestExtractDetectedLanguageFromResult(unittest.TestCase):
    """Test the _extract_detected_language_from_result function."""

    def test_none_result_returns_none(self):
        self.assertIsNone(_extract_detected_language_from_result(None))

    def test_object_with_language_attr(self):
        result = MagicMock()
        result.language = "fr"
        result.info = None
        lang = _extract_detected_language_from_result(result)
        self.assertEqual(lang, "fr")

    def test_dict_with_language_key(self):
        lang = _extract_detected_language_from_result({"language": "de"})
        self.assertEqual(lang, "de")

    def test_dict_with_lang_key(self):
        lang = _extract_detected_language_from_result({"lang": "es"})
        self.assertEqual(lang, "es")

    def test_dict_with_detected_language_key(self):
        lang = _extract_detected_language_from_result({"detected_language": "ja"})
        self.assertEqual(lang, "ja")

    def test_dict_with_info_dict(self):
        lang = _extract_detected_language_from_result({"info": {"language": "zh"}})
        self.assertEqual(lang, "zh")

    def test_empty_dict_returns_none(self):
        lang = _extract_detected_language_from_result({})
        self.assertIsNone(lang)

    def test_list_of_segments_with_language(self):
        seg = MagicMock()
        seg.language = "ko"
        lang = _extract_detected_language_from_result([seg])
        self.assertEqual(lang, "ko")

    def test_list_of_segments_with_dict(self):
        lang = _extract_detected_language_from_result([{"language": "ru"}])
        self.assertEqual(lang, "ru")

    def test_tuple_with_info_object(self):
        info = MagicMock()
        info.language = "it"
        lang = _extract_detected_language_from_result(("primary_result", info))
        self.assertEqual(lang, "it")

    def test_object_with_info_attr_with_language(self):
        info = MagicMock()
        info.language = "pt"
        result = MagicMock(spec=[])
        result.info = info
        # No .language attr on result itself, but has .info with .language
        lang = _extract_detected_language_from_result(result)
        self.assertEqual(lang, "pt")

    def test_list_with_dict_lang_key(self):
        lang = _extract_detected_language_from_result([{"lang": "ar"}])
        self.assertEqual(lang, "ar")


class TestGetEffectiveBackend(unittest.TestCase):
    """Test the get_effective_backend function."""

    def test_explicit_whispercpp_returns_whispercpp(self):
        result = get_effective_backend("whispercpp")
        self.assertEqual(result, "whispercpp")

    def test_explicit_faster_whisper_returns_faster_whisper(self):
        result = get_effective_backend("faster_whisper")
        self.assertEqual(result, "faster_whisper")

    def test_explicit_whisper_ctranslate2(self):
        result = get_effective_backend("whisper_ctranslate2")
        self.assertEqual(result, "whisper_ctranslate2")

    def test_whispercpp_cli_normalized(self):
        result = get_effective_backend("whispercpp_cli")
        self.assertEqual(result, "whispercpp")

    def test_hyphenated_backend_normalized(self):
        result = get_effective_backend("faster-whisper")
        self.assertEqual(result, "faster_whisper")

    def test_auto_returns_some_backend(self):
        result = get_effective_backend("auto")
        self.assertIn(result, ["whisper_ctranslate2", "whispercpp", "faster_whisper"])


class TestHallucinationPhrases(unittest.TestCase):
    """Test that HALLUCINATION_PHRASES is properly defined."""

    def test_is_list(self):
        from anytran.whisper_backend import HALLUCINATION_PHRASES
        self.assertIsInstance(HALLUCINATION_PHRASES, list)
        self.assertGreater(len(HALLUCINATION_PHRASES), 0)

    def test_all_lowercase(self):
        from anytran.whisper_backend import HALLUCINATION_PHRASES
        for phrase in HALLUCINATION_PHRASES:
            self.assertEqual(phrase, phrase.lower())


class TestModuleLevelConstants(unittest.TestCase):
    """Test module-level constants and variables."""

    def test_whisper_cpp_model_base_url(self):
        from anytran.whisper_backend import WHISPER_CPP_MODEL_BASE_URL
        self.assertIn("huggingface", WHISPER_CPP_MODEL_BASE_URL)

    def test_whisper_ctranslate2_available_is_bool(self):
        from anytran.whisper_backend import _whisper_ctranslate2_available
        self.assertIsInstance(_whisper_ctranslate2_available, bool)


if __name__ == "__main__":
    unittest.main()
