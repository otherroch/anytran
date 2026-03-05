"""Additional tests for anytran.text_translator — coverage of remaining branches."""
import unittest
from unittest.mock import MagicMock, patch

from tests.conftest import _real_text_translator_funcs as _TF

_translate_text = _TF["translate_text"]
_set_translation_backend = _TF["set_translation_backend"]
_get_translation_backend = _TF["get_translation_backend"]


class TestTranslateTextBranches(unittest.TestCase):
    """Test translate_text with various backends."""

    def test_empty_text_returns_as_is(self):
        result = _translate_text("", source_lang="en", target_lang="fr")
        self.assertEqual(result, "")

    def test_whitespace_only_returns_as_is(self):
        result = _translate_text("   ", source_lang="en", target_lang="fr")
        self.assertEqual(result, "   ")

    def test_same_language_no_translation(self):
        result = _translate_text("Hello", source_lang="en", target_lang="en")
        self.assertEqual(result, "Hello")

    def test_same_language_case_insensitive(self):
        result = _translate_text("Hello", source_lang="EN", target_lang="en")
        self.assertEqual(result, "Hello")

    def test_backend_none_passthrough(self):
        result = _translate_text("Hello", source_lang="en", target_lang="fr", backend="none")
        self.assertEqual(result, "Hello")

    def test_backend_passthrough_returns_text(self):
        result = _translate_text("Hello", source_lang="en", target_lang="fr", backend="passthrough")
        self.assertEqual(result, "Hello")

    def test_backend_unknown_returns_none(self):
        result = _translate_text("Hello", source_lang="en", target_lang="fr", backend="unknown_xyz", verbose=True)
        self.assertIsNone(result)

    def test_backend_metanllb_called(self):
        import anytran.text_translator as tt
        old_fn = _translate_text.__globals__.get("translate_text_metanllb")
        try:
            _translate_text.__globals__["translate_text_metanllb"] = lambda t, sl, tl, verbose=False: "Hola"
            result = _translate_text("Hello", source_lang="en", target_lang="es", backend="metanllb")
            self.assertEqual(result, "Hola")
        finally:
            if old_fn is not None:
                _translate_text.__globals__["translate_text_metanllb"] = old_fn

    def test_backend_marianmt_called(self):
        import anytran.text_translator as tt
        old_fn = _translate_text.__globals__.get("translate_text_marianmt")
        try:
            _translate_text.__globals__["translate_text_marianmt"] = lambda t, sl, tl, verbose=False: "Hola"
            result = _translate_text("Hello", source_lang="en", target_lang="es", backend="marianmt")
            self.assertEqual(result, "Hola")
        finally:
            if old_fn is not None:
                _translate_text.__globals__["translate_text_marianmt"] = old_fn

    def test_backend_translategemma_called(self):
        import anytran.text_translator as tt
        old_fn = _translate_text.__globals__.get("translate_text_translategemma")
        try:
            _translate_text.__globals__["translate_text_translategemma"] = lambda t, sl, tl, verbose=False: "Hola"
            result = _translate_text("Hello", source_lang="en", target_lang="es", backend="translategemma")
            self.assertEqual(result, "Hola")
        finally:
            if old_fn is not None:
                _translate_text.__globals__["translate_text_translategemma"] = old_fn

    def test_backend_googletrans_called(self):
        import anytran.text_translator as tt
        old_fn = _translate_text.__globals__.get("translate_text_googletrans")
        try:
            _translate_text.__globals__["translate_text_googletrans"] = lambda t, sl, tl, verbose=False: "Bonjour"
            result = _translate_text("Hello", source_lang="en", target_lang="fr", backend="googletrans")
            self.assertEqual(result, "Bonjour")
        finally:
            if old_fn is not None:
                _translate_text.__globals__["translate_text_googletrans"] = old_fn

    def test_backend_libretranslate_called(self):
        import anytran.text_translator as tt
        old_fn = _translate_text.__globals__.get("translate_text_libretranslate")
        try:
            _translate_text.__globals__["translate_text_libretranslate"] = lambda t, sl, tl, verbose=False: "Bonjour"
            result = _translate_text("Hello", source_lang="en", target_lang="fr", backend="libretranslate")
            self.assertEqual(result, "Bonjour")
        finally:
            if old_fn is not None:
                _translate_text.__globals__["translate_text_libretranslate"] = old_fn


class TestTranslationBackendConfig(unittest.TestCase):
    """Test translation backend configuration functions."""

    def setUp(self):
        self._orig_backend = _get_translation_backend()

    def tearDown(self):
        _set_translation_backend(self._orig_backend)

    def test_set_and_get_translation_backend(self):
        _set_translation_backend("libretranslate")
        self.assertEqual(_get_translation_backend(), "libretranslate")

    def test_default_backend_is_string(self):
        result = _get_translation_backend()
        self.assertIsInstance(result, str)


class TestTranslateTextGoogletrans(unittest.TestCase):
    """Test translate_text_googletrans with mocked translator."""

    def setUp(self):
        from tests.conftest import _real_text_translator_funcs as TF
        self._fn = TF["translate_text_googletrans"]

    def test_googletrans_when_not_available_returns_none(self):
        old_val = self._fn.__globals__.get("_GOOGLETRANS_AVAILABLE")
        try:
            self._fn.__globals__["_GOOGLETRANS_AVAILABLE"] = False
            result = self._fn("Hello", "en", "fr")
        finally:
            self._fn.__globals__["_GOOGLETRANS_AVAILABLE"] = old_val
        self.assertIsNone(result)

    def test_googletrans_with_mocked_translator(self):
        old_val = self._fn.__globals__.get("_GOOGLETRANS_AVAILABLE")
        old_getter = self._fn.__globals__.get("_get_googletrans_translator")
        try:
            mock_result = MagicMock()
            mock_result.text = "Bonjour"
            mock_translator = MagicMock()
            mock_translator.translate = MagicMock(return_value=mock_result)
            self._fn.__globals__["_GOOGLETRANS_AVAILABLE"] = True
            self._fn.__globals__["_get_googletrans_translator"] = lambda: mock_translator
            result = self._fn("Hello", "en", "fr")
        finally:
            self._fn.__globals__["_GOOGLETRANS_AVAILABLE"] = old_val
            self._fn.__globals__["_get_googletrans_translator"] = old_getter
        self.assertEqual(result, "Bonjour")

    def test_googletrans_exception_returns_none_after_retries(self):
        old_val = self._fn.__globals__.get("_GOOGLETRANS_AVAILABLE")
        old_getter = self._fn.__globals__.get("_get_googletrans_translator")
        try:
            mock_translator = MagicMock()
            mock_translator.translate = MagicMock(side_effect=Exception("API Error"))
            self._fn.__globals__["_GOOGLETRANS_AVAILABLE"] = True
            self._fn.__globals__["_get_googletrans_translator"] = lambda: mock_translator
            result = self._fn("Hello", "en", "fr", verbose=True)
        finally:
            self._fn.__globals__["_GOOGLETRANS_AVAILABLE"] = old_val
            self._fn.__globals__["_get_googletrans_translator"] = old_getter
        self.assertIsNone(result)


class TestSetTranslatorConfigs(unittest.TestCase):
    """Test set_*_config functions (these are mocked by test_looptran.py, so call from module directly)."""

    def test_set_libretranslate_config(self):
        """Just verify the function exists (may be mocked in test session)."""
        import anytran.text_translator as tt
        # We can't rely on the real function due to test_looptran.py mocking
        # but we can check the module has these attributes
        self.assertTrue(hasattr(tt, "set_libretranslate_config"))

    def test_set_translategemma_config(self):
        import anytran.text_translator as tt
        self.assertTrue(hasattr(tt, "set_translategemma_config"))

    def test_set_metanllb_config(self):
        import anytran.text_translator as tt
        self.assertTrue(hasattr(tt, "set_metanllb_config"))

    def test_set_marianmt_config(self):
        import anytran.text_translator as tt
        self.assertTrue(hasattr(tt, "set_marianmt_config"))


if __name__ == "__main__":
    unittest.main()
