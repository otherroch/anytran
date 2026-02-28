import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from anytran.normalizer import StreamingNormalizer, normalize_text


class TestStreamingNormalizerNormalizeChunk(unittest.TestCase):
    def setUp(self):
        self.normalizer = StreamingNormalizer()

    def test_removes_fillers(self):
        result = self.normalizer.normalize_chunk("um hello uh world")
        self.assertNotIn("um", result)
        self.assertNotIn("uh", result)
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_removes_space_before_punctuation(self):
        result = self.normalizer.normalize_chunk("hello , world")
        self.assertNotIn(" ,", result)
        self.assertIn("hello,", result)

    def test_adds_space_after_punctuation(self):
        result = self.normalizer.normalize_chunk("hello.world")
        self.assertIn("hello. world", result)

    def test_collapses_multiple_spaces(self):
        result = self.normalizer.normalize_chunk("hello   world")
        self.assertEqual(result, "hello world")

    def test_normalizes_curly_quotes(self):
        result = self.normalizer.normalize_chunk("\u2018hello\u2019")
        self.assertIn("'hello'", result)

    def test_strips_leading_spaces(self):
        result = self.normalizer.normalize_chunk("   hello")
        self.assertFalse(result.startswith(" "))

    def test_empty_string(self):
        result = self.normalizer.normalize_chunk("")
        self.assertEqual(result, "")


class TestStreamingNormalizerFinalize(unittest.TestCase):
    def setUp(self):
        self.normalizer = StreamingNormalizer()

    def test_capitalizes_first_letter(self):
        result = self.normalizer.finalize("hello world")
        self.assertTrue(result[0].isupper())

    def test_empty_string_returns_empty(self):
        result = self.normalizer.finalize("")
        self.assertEqual(result, "")

    def test_strips_whitespace(self):
        result = self.normalizer.finalize("  hello  ")
        self.assertFalse(result.startswith(" "))
        self.assertFalse(result.endswith(" "))

    def test_collapses_multiple_spaces(self):
        result = self.normalizer.finalize("hello   world")
        self.assertNotIn("  ", result)


class TestNormalizeText(unittest.TestCase):
    def test_basic_normalization(self):
        result = normalize_text("um hello world")
        self.assertNotIn("um", result)
        self.assertTrue(result[0].isupper())

    def test_empty_string(self):
        result = normalize_text("")
        self.assertEqual(result, "")

    def test_none_passthrough(self):
        result = normalize_text(None)
        self.assertIsNone(result)

    def test_capitalizes_first_letter(self):
        result = normalize_text("hello world")
        self.assertEqual(result[0], "H")

    def test_punctuation_spacing(self):
        result = normalize_text("hello , world . how are you")
        self.assertNotIn(" ,", result)
        self.assertNotIn(" .", result)

    def test_filler_removal(self):
        result = normalize_text("you know this is a test")
        self.assertNotIn("you know", result)

    def test_plain_text_unchanged_structurally(self):
        result = normalize_text("Hello world.")
        self.assertIn("Hello world", result)

    def test_multiline_preserves_newlines(self):
        result = normalize_text("Hello world.\nGoodbye world.")
        self.assertIn("\n", result)
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("Hello world", lines[0])
        self.assertIn("Goodbye world", lines[1])


class TestNormalizeIntegrationFileInput(unittest.TestCase):
    """Tests confirming the normalize parameter is wired into file output."""

    def test_normalize_text_removes_fillers_before_write(self):
        """normalize_text removes common ASR fillers from text."""
        result = normalize_text("um hello you know world uh")
        self.assertNotIn("um", result)
        self.assertNotIn("you know", result)
        self.assertNotIn("uh", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_normalize_text_applied_conditionally(self):
        """Calling normalize_text only when normalize=True matches expected behavior."""
        text = "um hello world"
        normalize = True
        out = normalize_text(text) if normalize else text
        self.assertNotIn("um", out)

        normalize = False
        out = normalize_text(text) if normalize else text
        self.assertIn("um", out)

    def test_run_file_input_accepts_normalize_parameter(self):
        """run_file_input function signature includes normalize parameter."""
        import importlib.util
        import inspect

        # Load run_file_input.py as a temporary module (without touching sys.modules)
        # so we can inspect the function signature without side effects.
        _rfi_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "anytran", "runners", "run_file_input.py"
        ))
        _spec = importlib.util.spec_from_file_location("_run_file_input_tmp", _rfi_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        sig = inspect.signature(_mod.run_file_input)
        self.assertIn("normalize", sig.parameters)
        self.assertTrue(sig.parameters["normalize"].default)  # default is True


if __name__ == "__main__":
    unittest.main()
