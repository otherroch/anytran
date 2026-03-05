"""Additional tests for anytran.voice_table to increase coverage."""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestLoadExistingEntries(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        from anytran.voice_table import load_existing_entries
        result = load_existing_entries("/tmp/nonexistent_voice_table.json")
        self.assertEqual(result, [])

    def test_valid_file_returns_list(self):
        from anytran.voice_table import load_existing_entries
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"onnx_file": "test.onnx"}], f)
            fname = f.name
        try:
            result = load_existing_entries(fname)
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(fname)

    def test_invalid_json_returns_empty(self):
        from anytran.voice_table import load_existing_entries
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            fname = f.name
        try:
            result = load_existing_entries(fname)
            self.assertEqual(result, [])
        finally:
            os.unlink(fname)

    def test_non_list_json_returns_empty(self):
        from anytran.voice_table import load_existing_entries
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            fname = f.name
        try:
            result = load_existing_entries(fname)
            self.assertEqual(result, [])
        finally:
            os.unlink(fname)


class TestAppendUniqueEntries(unittest.TestCase):
    def test_adds_new_entry_with_onnx_file(self):
        from anytran.voice_table import append_unique_entries
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            fname = f.name
        try:
            result = append_unique_entries(fname, [{"onnx_file": "test.onnx", "pitch": 150}])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["onnx_file"], "test.onnx")
        finally:
            os.unlink(fname)

    def test_no_duplicate_entries(self):
        from anytran.voice_table import append_unique_entries
        entry = {"onnx_file": "test.onnx", "pitch": 150}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([entry], f)
            fname = f.name
        try:
            result = append_unique_entries(fname, [entry])
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(fname)

    def test_entry_without_onnx_file_ignored(self):
        from anytran.voice_table import append_unique_entries
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            fname = f.name
        try:
            result = append_unique_entries(fname, [{"pitch": 150}])
            self.assertEqual(len(result), 0)
        finally:
            os.unlink(fname)


class TestGetSelectedLanguages(unittest.TestCase):
    def test_single_language(self):
        from anytran.voice_table import get_selected_languages
        result = get_selected_languages("en")
        self.assertIn("en", result)

    def test_comma_separated(self):
        from anytran.voice_table import get_selected_languages
        result = get_selected_languages("en,fr")
        self.assertIn("en", result)
        self.assertIn("fr", result)

    def test_all_returns_all(self):
        from anytran.voice_table import get_selected_languages
        result = get_selected_languages("all")
        self.assertEqual(result, ["all"])

    def test_empty_string_returns_default_fr(self):
        from anytran.voice_table import get_selected_languages
        result = get_selected_languages("")
        self.assertEqual(result, ["fr"])


class TestCollectVoicesForLanguages(unittest.TestCase):
    def test_empty_catalog_returns_empty(self):
        from anytran.voice_table import collect_voices_for_languages
        voices, unknowns = collect_voices_for_languages({}, ["en"])
        self.assertEqual(voices, [])

    def test_matching_language_returns_voices(self):
        from anytran.voice_table import collect_voices_for_languages
        catalog = {
            "en_US-test-medium": {
                "language": {"family": "en", "code": "en_us"},
                "files": {"en/en_US/en_US-test-medium/en_US-test-medium.onnx": {}},
            }
        }
        voices, unknowns = collect_voices_for_languages(catalog, ["en"])
        self.assertGreater(len(voices), 0)

    def test_unknown_language_reported(self):
        from anytran.voice_table import collect_voices_for_languages
        voices, unknowns = collect_voices_for_languages({}, ["xx_fake"])
        self.assertIn("xx_fake", unknowns)

    def test_all_returns_all_voices(self):
        from anytran.voice_table import collect_voices_for_languages
        catalog = {
            "en_US-test": {
                "language": {"family": "en", "code": "en_us"},
                "files": {"path/to/en_US-test.onnx": {}},
            },
            "fr_FR-test": {
                "language": {"family": "fr", "code": "fr_fr"},
                "files": {"path/to/fr_FR-test.onnx": {}},
            },
        }
        voices, unknowns = collect_voices_for_languages(catalog, ["all"])
        self.assertGreater(len(voices), 0)


if __name__ == "__main__":
    unittest.main()



if __name__ == '__main__': unittest.main()
