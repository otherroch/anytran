"""Tests for anytran.vad module."""
import unittest
from unittest.mock import MagicMock, patch


class TestVadModuleImport(unittest.TestCase):
    def test_silero_available_flag_is_bool(self):
        from anytran.vad import SILERO_AVAILABLE
        self.assertIsInstance(SILERO_AVAILABLE, bool)


class TestHasSpeechSileroWithoutSilero(unittest.TestCase):
    """Test has_speech_silero when silero is not available."""

    def setUp(self):
        import anytran.vad as vad_module
        self._orig_available = vad_module.SILERO_AVAILABLE
        vad_module.SILERO_AVAILABLE = False

    def tearDown(self):
        import anytran.vad as vad_module
        vad_module.SILERO_AVAILABLE = self._orig_available

    def test_returns_none_when_unavailable(self):
        import numpy as np
        from anytran.vad import has_speech_silero
        audio = np.zeros(16000, dtype=np.float32)
        result = has_speech_silero(audio, sample_rate=16000)
        self.assertIsNone(result)


class TestGetVadModelWithoutSilero(unittest.TestCase):
    """Test get_vad_model when silero is not available."""

    def setUp(self):
        import anytran.vad as vad_module
        self._orig_available = vad_module.SILERO_AVAILABLE
        vad_module.SILERO_AVAILABLE = False
        self._orig_model = vad_module._vad_model
        vad_module._vad_model = None

    def tearDown(self):
        import anytran.vad as vad_module
        vad_module.SILERO_AVAILABLE = self._orig_available
        vad_module._vad_model = self._orig_model

    def test_returns_none_when_unavailable(self):
        from anytran.vad import get_vad_model
        result = get_vad_model()
        self.assertIsNone(result)


class TestHasSpeechSileroWithMockedSilero(unittest.TestCase):
    """Test has_speech_silero when silero is mocked as available."""

    def setUp(self):
        import anytran.vad as vad_module
        self._orig_available = vad_module.SILERO_AVAILABLE
        self._orig_model = vad_module._vad_model

    def tearDown(self):
        import anytran.vad as vad_module
        vad_module.SILERO_AVAILABLE = self._orig_available
        vad_module._vad_model = self._orig_model

    def test_has_speech_returns_true_when_timestamps_found(self):
        import numpy as np
        import anytran.vad as vad_module

        mock_vad = MagicMock()
        vad_module._vad_model = mock_vad
        vad_module.SILERO_AVAILABLE = True

        mock_get_speech = MagicMock(return_value=[{"start": 0, "end": 100}])
        # Inject the names into the module if not already there
        vad_module.get_speech_timestamps = mock_get_speech
        with patch("anytran.vad.get_vad_model", return_value=mock_vad):
            result = vad_module.has_speech_silero(
                np.zeros(16000, dtype=np.float32), sample_rate=16000
            )
        self.assertTrue(result)

    def test_has_speech_returns_false_when_no_timestamps(self):
        import numpy as np
        import anytran.vad as vad_module

        mock_vad = MagicMock()
        vad_module._vad_model = mock_vad
        vad_module.SILERO_AVAILABLE = True

        mock_get_speech = MagicMock(return_value=[])
        vad_module.get_speech_timestamps = mock_get_speech
        with patch("anytran.vad.get_vad_model", return_value=mock_vad):
            result = vad_module.has_speech_silero(
                np.zeros(16000, dtype=np.float32), sample_rate=16000
            )
        self.assertFalse(result)

    def test_has_speech_returns_none_when_vad_model_none(self):
        import numpy as np
        import anytran.vad as vad_module

        vad_module.SILERO_AVAILABLE = True

        with patch("anytran.vad.get_vad_model", return_value=None):
            result = vad_module.has_speech_silero(
                np.zeros(16000, dtype=np.float32), sample_rate=16000
            )
        self.assertIsNone(result)

    def test_has_speech_returns_none_on_exception(self):
        import numpy as np
        import anytran.vad as vad_module

        mock_vad = MagicMock()
        vad_module.SILERO_AVAILABLE = True

        mock_get_speech = MagicMock(side_effect=RuntimeError("fail"))
        vad_module.get_speech_timestamps = mock_get_speech
        with patch("anytran.vad.get_vad_model", return_value=mock_vad):
            result = vad_module.has_speech_silero(
                np.zeros(16000, dtype=np.float32), sample_rate=16000
            )
        self.assertIsNone(result)

    def test_get_vad_model_loads_when_available(self):
        import anytran.vad as vad_module

        vad_module.SILERO_AVAILABLE = True
        vad_module._vad_model = None
        mock_model = MagicMock()

        # Inject load_silero_vad into the module namespace
        vad_module.load_silero_vad = MagicMock(return_value=mock_model)
        result = vad_module.get_vad_model()
        self.assertEqual(result, mock_model)
        vad_module._vad_model = None  # reset


if __name__ == "__main__":
    unittest.main()
