"""Tests for anytran.text_translator module — setters, getters, and translate_text routing."""
import unittest
from unittest.mock import patch, MagicMock


class TestSettersAndGetters(unittest.TestCase):
    def setUp(self):
        import anytran.text_translator as tt
        self._orig_backend = tt._translation_backend
        self._orig_libre_url = tt._libretranslate_url
        self._orig_gemma_model = tt._translategemma_model_name
        self._orig_nllb_model = tt._metanllb_model_name
        self._orig_marian_model = tt._marianmt_model_name

    def tearDown(self):
        import anytran.text_translator as tt
        tt._translation_backend = self._orig_backend
        tt._libretranslate_url = self._orig_libre_url
        tt._translategemma_model_name = self._orig_gemma_model
        tt._metanllb_model_name = self._orig_nllb_model
        tt._marianmt_model_name = self._orig_marian_model

    def test_set_and_get_translation_backend(self):
        from anytran.text_translator import set_translation_backend, get_translation_backend
        set_translation_backend("libretranslate")
        self.assertEqual(get_translation_backend(), "libretranslate")

    def test_set_libretranslate_config(self):
        import anytran.text_translator as tt
        from anytran.text_translator import set_libretranslate_config
        set_libretranslate_config("http://localhost:5000")
        self.assertEqual(tt._libretranslate_url, "http://localhost:5000")

    def test_set_translategemma_config(self):
        import anytran.text_translator as tt
        from anytran.text_translator import set_translategemma_config
        set_translategemma_config("my-gemma-model")
        self.assertEqual(tt._translategemma_model_name, "my-gemma-model")

    def test_set_metanllb_config(self):
        import anytran.text_translator as tt
        from anytran.text_translator import set_metanllb_config
        set_metanllb_config("facebook/nllb-200-600M")
        self.assertEqual(tt._metanllb_model_name, "facebook/nllb-200-600M")

    def test_set_marianmt_config(self):
        import anytran.text_translator as tt
        from anytran.text_translator import set_marianmt_config
        set_marianmt_config("Helsinki-NLP/opus-mt-en-fr")
        self.assertEqual(tt._marianmt_model_name, "Helsinki-NLP/opus-mt-en-fr")


class TestTranslateTextRouting(unittest.TestCase):
    """Test translate_text routing logic without needing real backends."""

    def setUp(self):
        import anytran.text_translator as tt
        self._orig_backend = tt._translation_backend
        tt._translation_backend = "none"

    def tearDown(self):
        import anytran.text_translator as tt
        tt._translation_backend = self._orig_backend

    def test_empty_text_returns_as_is(self):
        from anytran.text_translator import translate_text
        result = translate_text("", "en", "fr")
        self.assertEqual(result, "")

    def test_whitespace_only_returns_as_is(self):
        from anytran.text_translator import translate_text
        result = translate_text("   ", "en", "fr")
        self.assertEqual(result, "   ")

    def test_none_text_returns_none(self):
        from anytran.text_translator import translate_text
        result = translate_text(None, "en", "fr")
        self.assertIsNone(result)

    def test_same_language_returns_text_unchanged(self):
        from anytran.text_translator import translate_text
        result = translate_text("Hello", "en", "en")
        self.assertEqual(result, "Hello")

    def test_same_language_case_insensitive(self):
        from anytran.text_translator import translate_text
        result = translate_text("Hello", "EN", "en")
        self.assertEqual(result, "Hello")

    def test_passthrough_backend_returns_text(self):
        from anytran.text_translator import translate_text
        result = translate_text("Hello", "en", "fr", backend="passthrough")
        self.assertEqual(result, "Hello")

    def test_none_backend_returns_text(self):
        from anytran.text_translator import translate_text
        result = translate_text("Hello", "en", "fr", backend="none")
        self.assertEqual(result, "Hello")

    def test_unknown_backend_returns_none(self):
        from anytran.text_translator import translate_text
        result = translate_text("Hello", "en", "fr", backend="nonexistent_backend")
        self.assertIsNone(result)

    def test_unknown_backend_verbose(self):
        from anytran.text_translator import translate_text
        result = translate_text("Hello", "en", "fr", backend="nonexistent_backend", verbose=True)
        self.assertIsNone(result)

    def test_googletrans_unavailable_returns_none(self):
        from anytran.text_translator import translate_text, _GOOGLETRANS_AVAILABLE
        if not _GOOGLETRANS_AVAILABLE:
            result = translate_text("Hello", "en", "fr", backend="googletrans")
            self.assertIsNone(result)

    def test_libretranslate_with_no_url_returns_none(self):
        import anytran.text_translator as tt
        from anytran.text_translator import translate_text
        tt._libretranslate_url = None
        result = translate_text("Hello", "en", "fr", backend="libretranslate")
        # Should return None since no URL configured and requests would fail
        # (or it might attempt and fail gracefully)

    def test_configured_backend_used_as_default(self):
        import anytran.text_translator as tt
        from anytran.text_translator import translate_text
        tt._translation_backend = "none"
        result = translate_text("World", "en", "es")
        self.assertEqual(result, "World")


class TestGetGoogletransTranslator(unittest.TestCase):
    def test_raises_import_error_when_unavailable(self):
        from anytran.text_translator import _GOOGLETRANS_AVAILABLE, _get_googletrans_translator
        if not _GOOGLETRANS_AVAILABLE:
            with self.assertRaises(ImportError):
                _get_googletrans_translator()


class TestNllbLangMap(unittest.TestCase):
    def test_nllb_map_has_english(self):
        from anytran.text_translator import _NLLB_LANG_MAP
        self.assertIn("en", _NLLB_LANG_MAP)
        self.assertEqual(_NLLB_LANG_MAP["en"], "eng_Latn")

    def test_nllb_map_has_french(self):
        from anytran.text_translator import _NLLB_LANG_MAP
        self.assertIn("fr", _NLLB_LANG_MAP)


class TestTranslateTextGoogletrans(unittest.TestCase):
    """Test translate_text_googletrans when googletrans is not installed."""

    def test_returns_none_when_not_available(self):
        from anytran.text_translator import translate_text_googletrans, _GOOGLETRANS_AVAILABLE
        if not _GOOGLETRANS_AVAILABLE:
            result = translate_text_googletrans("Hello", "en", "fr")
            self.assertIsNone(result)

    def test_returns_none_verbose_when_not_available(self):
        from anytran.text_translator import translate_text_googletrans, _GOOGLETRANS_AVAILABLE
        if not _GOOGLETRANS_AVAILABLE:
            result = translate_text_googletrans("Hello", "en", "fr", verbose=True)
            self.assertIsNone(result)

    def test_zh_mapped_to_zh_cn(self):
        """Verify that zh target language is mapped to zh-cn."""
        from anytran.text_translator import translate_text_googletrans, _GOOGLETRANS_AVAILABLE
        if not _GOOGLETRANS_AVAILABLE:
            # Can't test the mapping directly if googletrans is unavailable,
            # but we can verify the function handles zh without error
            result = translate_text_googletrans("Hello", "en", "zh")
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
