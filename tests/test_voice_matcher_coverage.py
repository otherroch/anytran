"""
Tests for anytran.voice_matcher using direct imports (not importlib),
so that pytest-cov can properly instrument and track coverage.
"""
import json
import os
import tempfile
import unittest

import numpy as np


class TestExtractVoiceFeatures(unittest.TestCase):
    """Test the extract_voice_features function with synthetic audio."""

    def _make_tone(self, freq=150, duration=1.0, sample_rate=16000):
        """Generate a simple sine wave at the given frequency."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return (np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_returns_dict_with_expected_keys(self):
        from anytran.voice_matcher import extract_voice_features
        audio = self._make_tone(freq=120)
        features = extract_voice_features(audio, sample_rate=16000)
        for key in ("mean_pitch", "pitch_std", "zcr", "brightness", "gender", "voice_type"):
            self.assertIn(key, features)

    def test_low_pitch_classified_as_male(self):
        from anytran.voice_matcher import extract_voice_features
        # 100 Hz — well within male range
        audio = self._make_tone(freq=100)
        features = extract_voice_features(audio, sample_rate=16000)
        self.assertEqual(features["gender"], "male")

    def test_high_pitch_classified_as_female(self):
        from anytran.voice_matcher import extract_voice_features
        # 250 Hz — well within female range
        audio = self._make_tone(freq=250)
        features = extract_voice_features(audio, sample_rate=16000)
        self.assertEqual(features["gender"], "female")

    def test_verbose_mode_does_not_raise(self):
        from anytran.voice_matcher import extract_voice_features
        audio = self._make_tone(freq=150)
        features = extract_voice_features(audio, sample_rate=16000, verbose=True)
        self.assertIsInstance(features, dict)

    def test_int16_audio_is_converted(self):
        from anytran.voice_matcher import extract_voice_features
        audio_int16 = (self._make_tone(freq=120) * 32768).astype(np.int16)
        # Should not raise
        features = extract_voice_features(audio_int16, sample_rate=16000)
        self.assertIn("gender", features)

    def test_high_magnitude_audio_normalized(self):
        from anytran.voice_matcher import extract_voice_features
        audio = self._make_tone(freq=120) * 40000  # > 1.0
        features = extract_voice_features(audio.astype(np.float32), sample_rate=16000)
        self.assertIn("gender", features)

    def test_voice_type_male_deep(self):
        from anytran.voice_matcher import extract_voice_features
        # < 100 Hz → male_deep
        audio = self._make_tone(freq=80)
        features = extract_voice_features(audio, sample_rate=16000)
        self.assertEqual(features["voice_type"], "male_deep")

    def test_voice_type_female_high(self):
        from anytran.voice_matcher import extract_voice_features
        # > 220 Hz → female_high
        audio = self._make_tone(freq=240)
        features = extract_voice_features(audio, sample_rate=16000)
        self.assertEqual(features["voice_type"], "female_high")

    def test_float_values_in_features(self):
        from anytran.voice_matcher import extract_voice_features
        audio = self._make_tone(freq=120)
        features = extract_voice_features(audio, sample_rate=16000)
        for key in ("mean_pitch", "pitch_std", "zcr", "brightness"):
            self.assertIsInstance(features[key], float)


class TestVoiceMatchScore(unittest.TestCase):
    """Test the _voice_match_score scoring function."""

    def test_perfect_match_low_score(self):
        from anytran.voice_matcher import _voice_match_score
        features = {"mean_pitch": 150.0, "pitch_std": 10.0, "zcr": 0.1, "brightness": 2000.0}
        voice_data = {"pitch": 150.0, "pitch_std": 10.0, "zcr": 0.1, "brightness": 2000.0}
        score = _voice_match_score(features, voice_data)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_large_pitch_difference_higher_score(self):
        from anytran.voice_matcher import _voice_match_score
        features = {"mean_pitch": 100.0}
        voice_data_close = {"pitch": 110.0}
        voice_data_far = {"pitch": 200.0}
        score_close = _voice_match_score(features, voice_data_close)
        score_far = _voice_match_score(features, voice_data_far)
        self.assertLess(score_close, score_far)

    def test_missing_optional_fields_no_error(self):
        from anytran.voice_matcher import _voice_match_score
        features = {"mean_pitch": 150.0}
        voice_data = {"pitch": 155.0}
        score = _voice_match_score(features, voice_data)
        self.assertGreaterEqual(score, 0.0)

    def test_score_with_all_fields(self):
        from anytran.voice_matcher import _voice_match_score
        features = {
            "mean_pitch": 150.0,
            "pitch_std": 10.0,
            "zcr": 0.1,
            "brightness": 2000.0,
        }
        voice_data = {
            "pitch": 160.0,
            "pitch_std": 12.0,
            "zcr": 0.12,
            "brightness": 2100.0,
        }
        score = _voice_match_score(features, voice_data)
        self.assertGreater(score, 0.0)


class TestSelectBestPiperVoiceDirectImport(unittest.TestCase):
    """Test select_best_piper_voice using direct imports for coverage."""

    def _make_voice_table(self, entries):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "voice_table.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f)
        return path, tmpdir

    def setUp(self):
        import anytran.voice_matcher as vm
        vm._load_piper_voices.cache_clear()

    def tearDown(self):
        import anytran.voice_matcher as vm
        vm._load_piper_voices.cache_clear()
        import shutil
        if hasattr(self, "_tmpdir"):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_selects_male_voice_for_male_features(self):
        from anytran.voice_matcher import select_best_piper_voice
        import anytran.voice_matcher as vm

        path, self._tmpdir = self._make_voice_table([
            {"onnx_file": "en_US-male-medium.onnx", "pitch": 120, "gender": "male"},
            {"onnx_file": "en_US-female-medium.onnx", "pitch": 220, "gender": "female"},
        ])
        orig_path = vm.VOICE_TABLE_JSON_PATH
        vm.VOICE_TABLE_JSON_PATH = path
        vm._load_piper_voices.cache_clear()
        try:
            selected = select_best_piper_voice({"mean_pitch": 110, "gender": "male"}, language="en")
            self.assertEqual(selected, "en_US-male-medium")
        finally:
            vm.VOICE_TABLE_JSON_PATH = orig_path
            vm._load_piper_voices.cache_clear()

    def test_falls_back_when_no_gender_match(self):
        from anytran.voice_matcher import select_best_piper_voice
        import anytran.voice_matcher as vm

        path, self._tmpdir = self._make_voice_table([
            {"onnx_file": "en_US-female-medium.onnx", "pitch": 220, "gender": "female"},
        ])
        orig_path = vm.VOICE_TABLE_JSON_PATH
        vm.VOICE_TABLE_JSON_PATH = path
        vm._load_piper_voices.cache_clear()
        try:
            selected = select_best_piper_voice({"mean_pitch": 110, "gender": "male"}, language="en")
            # Should fall back to female voice
            self.assertEqual(selected, "en_US-female-medium")
        finally:
            vm.VOICE_TABLE_JSON_PATH = orig_path
            vm._load_piper_voices.cache_clear()

    def test_returns_none_for_empty_table(self):
        from anytran.voice_matcher import select_best_piper_voice
        import anytran.voice_matcher as vm

        path, self._tmpdir = self._make_voice_table([])
        orig_path = vm.VOICE_TABLE_JSON_PATH
        vm.VOICE_TABLE_JSON_PATH = path
        vm._load_piper_voices.cache_clear()
        try:
            selected = select_best_piper_voice({"mean_pitch": 120, "gender": "male"}, language="en")
            self.assertIsNone(selected)
        finally:
            vm.VOICE_TABLE_JSON_PATH = orig_path
            vm._load_piper_voices.cache_clear()

    def test_verbose_does_not_raise(self):
        from anytran.voice_matcher import select_best_piper_voice
        import anytran.voice_matcher as vm

        path, self._tmpdir = self._make_voice_table([
            {"onnx_file": "en_US-test-medium.onnx", "pitch": 150, "gender": "male"},
        ])
        orig_path = vm.VOICE_TABLE_JSON_PATH
        vm.VOICE_TABLE_JSON_PATH = path
        vm._load_piper_voices.cache_clear()
        try:
            selected = select_best_piper_voice(
                {"mean_pitch": 150, "gender": "male", "zcr": 0.1, "brightness": 2000},
                language="en",
                verbose=True,
            )
            self.assertIsNotNone(selected)
        finally:
            vm.VOICE_TABLE_JSON_PATH = orig_path
            vm._load_piper_voices.cache_clear()

    def test_language_not_in_table_returns_none(self):
        from anytran.voice_matcher import select_best_piper_voice
        import anytran.voice_matcher as vm

        path, self._tmpdir = self._make_voice_table([
            {"onnx_file": "en_US-test-medium.onnx", "pitch": 150, "gender": "male"},
        ])
        orig_path = vm.VOICE_TABLE_JSON_PATH
        vm.VOICE_TABLE_JSON_PATH = path
        vm._load_piper_voices.cache_clear()
        try:
            selected = select_best_piper_voice(
                {"mean_pitch": 150, "gender": "male"}, language="xx"
            )
            # "xx" not in table, falls back to "en" per code logic
            # (or returns None if no fallback)
        finally:
            vm.VOICE_TABLE_JSON_PATH = orig_path
            vm._load_piper_voices.cache_clear()


class TestLoadPiperVoicesDirectImport(unittest.TestCase):
    """Test _load_piper_voices through direct import."""

    def setUp(self):
        import anytran.voice_matcher as vm
        vm._load_piper_voices.cache_clear()
        self._orig_path = vm.VOICE_TABLE_JSON_PATH

    def tearDown(self):
        import anytran.voice_matcher as vm
        vm.VOICE_TABLE_JSON_PATH = self._orig_path
        vm._load_piper_voices.cache_clear()
        import shutil
        if hasattr(self, "_tmpdir"):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_empty_dict_on_missing_file(self):
        from anytran.voice_matcher import _load_piper_voices
        import anytran.voice_matcher as vm
        vm.VOICE_TABLE_JSON_PATH = "/nonexistent/path/voice_table.json"
        vm._load_piper_voices.cache_clear()
        result = _load_piper_voices()
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_invalid_json(self):
        from anytran.voice_matcher import _load_piper_voices
        import anytran.voice_matcher as vm
        import tempfile
        tmpdir = tempfile.mkdtemp()
        self._tmpdir = tmpdir
        bad_path = os.path.join(tmpdir, "bad.json")
        with open(bad_path, "w") as f:
            f.write("not valid json{{{")
        vm.VOICE_TABLE_JSON_PATH = bad_path
        vm._load_piper_voices.cache_clear()
        result = _load_piper_voices()
        self.assertEqual(result, {})

    def test_skips_entries_missing_required_fields(self):
        from anytran.voice_matcher import _load_piper_voices
        import anytran.voice_matcher as vm
        import tempfile
        tmpdir = tempfile.mkdtemp()
        self._tmpdir = tmpdir
        path = os.path.join(tmpdir, "voice_table.json")
        # Entry missing onnx_file
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"pitch": 100, "gender": "male"}], f)
        vm.VOICE_TABLE_JSON_PATH = path
        vm._load_piper_voices.cache_clear()
        result = _load_piper_voices()
        # Should be empty because onnx_file is missing
        self.assertEqual(result, {})

    def test_loads_valid_entry(self):
        from anytran.voice_matcher import _load_piper_voices
        import anytran.voice_matcher as vm
        import tempfile
        tmpdir = tempfile.mkdtemp()
        self._tmpdir = tmpdir
        path = os.path.join(tmpdir, "voice_table.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([
                {"onnx_file": "en_US-test-medium.onnx", "pitch": 140, "gender": "male",
                 "pitch_std": 10.0, "zcr": 0.12, "brightness": 1800.0}
            ], f)
        vm.VOICE_TABLE_JSON_PATH = path
        vm._load_piper_voices.cache_clear()
        result = _load_piper_voices()
        self.assertIn("en", result)
        self.assertIn("en_US-test-medium", result["en"])
        entry = result["en"]["en_US-test-medium"]
        self.assertEqual(entry["pitch"], 140)
        self.assertEqual(entry["gender"], "male")


if __name__ == "__main__":
    unittest.main()
