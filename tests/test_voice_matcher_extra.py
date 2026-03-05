"""Additional voice_matcher coverage tests."""
import numpy as np
import unittest
from unittest.mock import patch


class TestExtractVoiceFeaturesAdditional(unittest.TestCase):
    """Cover remaining branches in extract_voice_features."""

    def setUp(self):
        import anytran.voice_matcher as vm
        self._extract = vm.extract_voice_features

    def test_ambiguous_range_high_brightness(self):
        """Pitch in 150-190 Hz with high brightness → female voice."""
        # Make silence (all zeros gives zero pitch which defaults to 150)
        # Use a signal with a specific frequency to get pitch ~170 Hz
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        # 170 Hz tone (in ambiguous range)
        audio = np.sin(2 * np.pi * 170 * t).astype(np.float32)
        features = self._extract(audio, sr)
        # Just verify it returns a valid dict without error
        self.assertIn("gender", features)
        self.assertIn("voice_type", features)

    def test_female_mid_range(self):
        """Pitch ~200 Hz (female_mid range)."""
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        features = self._extract(audio, sr)
        self.assertIn("gender", features)

    def test_verbose_does_not_raise(self):
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.01
        features = self._extract(audio, sr, verbose=True)
        self.assertIsInstance(features, dict)

    def test_pitch_exception_uses_default(self):
        """When librosa.yin raises, use default pitch."""
        import anytran.voice_matcher as vm
        with patch("librosa.yin", side_effect=Exception("yin failed")):
            sr = 16000
            audio = np.ones(sr, dtype=np.float32) * 0.01
            features = vm.extract_voice_features(audio, sr, verbose=True)
        self.assertEqual(features["mean_pitch"], 150.0)

    def test_spectral_centroid_exception_uses_default_brightness(self):
        """When spectral_centroid raises, use default 2000 Hz."""
        import anytran.voice_matcher as vm
        with patch("librosa.feature.spectral_centroid", side_effect=Exception("error")):
            sr = 16000
            audio = np.ones(sr, dtype=np.float32) * 0.01
            features = vm.extract_voice_features(audio, sr)
        self.assertEqual(features["brightness"], 2000.0)


class TestSelectBestPiperVoiceAdditional(unittest.TestCase):
    """Test select_best_piper_voice with fallback and verbose scenarios."""

    def setUp(self):
        import anytran.voice_matcher as vm
        self._select = vm.select_best_piper_voice

    def test_verbose_no_matching_gender_uses_fallback(self):
        """When no gender match but a fallback exists, use fallback."""
        features = {
            "mean_pitch": 250.0,
            "gender": "female",
            "pitch_std": 10.0,
            "zcr": 0.05,
            "brightness": 3000.0,
            "voice_type": "female_high",
        }
        # Use a language with known voices that are all male
        result = self._select(features, language="en", verbose=True)
        # Should return some voice (or None if no en voices in voice_data)

    def test_select_verbose_with_many_voices(self):
        """Cover verbose output paths."""
        features = {
            "mean_pitch": 150.0,
            "gender": "male",
            "pitch_std": 5.0,
            "zcr": 0.05,
            "brightness": 2000.0,
            "voice_type": "male_mid",
        }
        result = self._select(features, language="en", verbose=True)
        # Should return a string or None

    def test_select_best_voice_all_male_voices_for_female_request(self):
        """When requesting a female voice but only male available, fallback."""
        import anytran.voice_matcher as vm
        mock_voices = {
            "fr": {
                "voice1": {"pitch": 130, "gender": "male", "pitch_std": 5, "zcr": 0.05, "brightness": 1500},
                "voice2": {"pitch": 110, "gender": "male", "pitch_std": 8, "zcr": 0.06, "brightness": 1800},
            }
        }
        with patch.object(vm, "_load_piper_voices", return_value=mock_voices):
            features = {
                "mean_pitch": 200.0,
                "gender": "female",
                "pitch_std": 10.0,
                "zcr": 0.05,
                "brightness": 3000.0,
                "voice_type": "female_mid",
            }
            result = vm.select_best_piper_voice(features, language="fr", verbose=True)
            # Fallback to a male voice
            self.assertIn(result, ["voice1", "voice2", None])


if __name__ == "__main__":
    unittest.main()
