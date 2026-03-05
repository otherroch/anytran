"""Tests for anytran.web_server — _serialize_tts_segments and run_web_server setup."""
import os
import unittest
from unittest.mock import MagicMock, patch

from tests.conftest import _real_web_server_funcs as _WSF

_run_web_server = _WSF["run_web_server"]
_serialize_tts_segments = _WSF["_serialize_tts_segments"]

try:
    import fastapi  # noqa: F401
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


class TestSerializeTtsSegments(unittest.TestCase):
    """Test the _serialize_tts_segments helper function."""

    def test_none_returns_none(self):
        _serialize_tts_segments = _WSF["_serialize_tts_segments"]  # already imported above
        self.assertIsNone(_serialize_tts_segments(None, 16000))

    def test_empty_list_returns_none(self):
        _serialize_tts_segments = _WSF["_serialize_tts_segments"]  # already imported above
        self.assertIsNone(_serialize_tts_segments([], 16000))

    def test_none_segment_skipped(self):
        _serialize_tts_segments = _WSF["_serialize_tts_segments"]  # already imported above
        result = _serialize_tts_segments([None], 16000)
        # None-only list: all segments skipped → empty payloads → returns []
        self.assertIsNotNone(result)

    def test_valid_segment_returns_list(self):
        import numpy as np
        _serialize_tts_segments = _WSF["_serialize_tts_segments"]  # already imported above
        audio = np.ones(1600, dtype=np.float32) * 0.1
        result = _serialize_tts_segments([audio], 16000)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_payload_has_expected_keys(self):
        import numpy as np
        _serialize_tts_segments = _WSF["_serialize_tts_segments"]  # already imported above
        audio = np.ones(1600, dtype=np.float32) * 0.1
        result = _serialize_tts_segments([audio], 16000)
        item = result[0]
        self.assertEqual(item["type"], "tts_audio")
        self.assertIn("pcm", item)
        self.assertEqual(item["rate"], 16000)

    def test_pcm_is_base64_string(self):
        import numpy as np
        import base64
        _serialize_tts_segments = _WSF["_serialize_tts_segments"]  # already imported above
        audio = np.ones(1600, dtype=np.float32) * 0.1
        result = _serialize_tts_segments([audio], 16000)
        pcm_b64 = result[0]["pcm"]
        self.assertIsInstance(pcm_b64, str)
        # Should be valid base64
        decoded = base64.b64decode(pcm_b64)
        self.assertGreater(len(decoded), 0)

    def test_multiple_segments(self):
        import numpy as np
        _serialize_tts_segments = _WSF["_serialize_tts_segments"]  # already imported above
        seg1 = np.ones(800, dtype=np.float32) * 0.1
        seg2 = np.ones(800, dtype=np.float32) * 0.2
        result = _serialize_tts_segments([seg1, seg2], 22050)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["rate"], 22050)


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed — skipping run_web_server tests")
class TestRunWebServerSetup(unittest.TestCase):
    """Test the run_web_server function setup code (before uvicorn.run is called)."""

    def test_run_web_server_creates_app(self):
        """Calling run_web_server with mocked uvicorn should not raise."""
        pass  # use _run_web_server from module level

        mock_server = MagicMock()
        mock_server.run = MagicMock()
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            _run_web_server(
                input_lang="en",
                output_lang="fr",
                host="127.0.0.1",
                port=8765,
                mqtt_broker=None,
            )

        mock_server.run.assert_called_once()

    def test_run_web_server_with_defaults(self):
        """run_web_server with all defaults should complete setup."""
        pass  # use _run_web_server from module level

        mock_server = MagicMock()
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            _run_web_server()

        mock_server.run.assert_called_once()

    def test_run_web_server_with_auto_input(self):
        """run_web_server with 'auto' input lang should set default_input_lang='auto'."""
        pass  # use _run_web_server from module level

        mock_server = MagicMock()
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            # "auto" input lang - shouldn't raise
            _run_web_server(input_lang=None, output_lang="en")

        mock_server.run.assert_called_once()

    def test_run_web_server_with_piper_env(self):
        """Test with USE_PIPER env var set."""
        pass  # use _run_web_server from module level

        mock_server = MagicMock()
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with patch.dict("os.environ", {"USE_PIPER": "1", "PIPER_VOICE": "en_US-test"}):
            with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
                _run_web_server(input_lang="fr", output_lang="en")

        mock_server.run.assert_called_once()

    def test_run_web_server_with_ssl(self):
        """Test with SSL cert/key provided."""
        pass  # use _run_web_server from module level
        import tempfile

        mock_server = MagicMock()
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as cf:
            cert_path = cf.name
        with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as kf:
            key_path = kf.name

        try:
            with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
                _run_web_server(
                    ssl_certfile=cert_path,
                    ssl_keyfile=key_path,
                )
            mock_server.run.assert_called_once()
        finally:
            os.unlink(cert_path)
            os.unlink(key_path)

    def test_run_web_server_with_mqtt_broker(self):
        """Test with mqtt_broker set - should call init_mqtt."""
        pass  # use _run_web_server from module level

        mock_server = MagicMock()
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with patch("anytran.web_server.init_mqtt") as mock_init_mqtt:
            with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
                _run_web_server(mqtt_broker="localhost", mqtt_port=1883)
            mock_init_mqtt.assert_called_once()

    def test_run_web_server_connection_reset_error_windows(self):
        """Test that ConnectionResetError during shutdown is handled on Windows."""
        pass  # use _run_web_server from module level

        mock_server = MagicMock()
        mock_server.run = MagicMock(side_effect=ConnectionResetError("WinError 10054"))
        mock_server.should_exit = True
        mock_server.force_exit = True
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            with patch("os.name", "nt"):
                # Should not raise on Windows when server is shutting down
                _run_web_server()

    def test_run_web_server_connection_reset_error_non_windows(self):
        """On non-Windows, ConnectionResetError should propagate."""
        pass  # use _run_web_server from module level

        mock_server = MagicMock()
        mock_server.run = MagicMock(side_effect=ConnectionResetError("error"))
        mock_server.should_exit = False
        mock_server.force_exit = False
        mock_uvicorn = MagicMock()
        mock_uvicorn.Config = MagicMock(return_value=MagicMock())
        mock_uvicorn.Server = MagicMock(return_value=mock_server)

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            with patch("os.name", "posix"):
                with self.assertRaises(ConnectionResetError):
                    _run_web_server()


if __name__ == "__main__":
    unittest.main()
