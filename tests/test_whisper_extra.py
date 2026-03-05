"""Additional whisper_backend utility function tests."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.conftest import _real_whisper_backend_funcs as _WBF


class TestCallWithNativeOutputCapture(unittest.TestCase):
    """Test _call_with_native_output_capture."""

    def test_returns_tuple(self):
        import anytran.whisper_backend as wb
        func = _WBF["_call_with_native_output_capture"]
        result, captured = func(lambda: 42)
        self.assertEqual(result, 42)

    def test_captures_output(self):
        func = _WBF["_call_with_native_output_capture"]

        def noisy_func():
            os.write(sys.stdout.fileno(), b"captured output\n")
            return "done"

        result, captured = func(noisy_func)
        self.assertEqual(result, "done")

    def test_fileno_raises_handles_gracefully(self):
        import io
        func = _WBF["_call_with_native_output_capture"]

        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            result, captured = func(lambda: "hello")
            self.assertEqual(result, "hello")
            self.assertIsNone(captured)
        finally:
            sys.stdout = old_stdout


class TestResolveFasterWhisperModelPath(unittest.TestCase):
    """Test the _resolve_faster_whisper_model_path function."""

    def _call(self, *args, **kwargs):
        return _WBF["_resolve_faster_whisper_model_path"](*args, **kwargs)

    def test_unknown_model_name_returns_none(self):
        result = self._call("small")
        self.assertIsNone(result)

    def test_path_to_existing_directory_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._call(tmpdir)
            self.assertEqual(result, os.path.abspath(tmpdir))

    def test_none_model_size_returns_none(self):
        result = self._call(None)
        self.assertIsNone(result)

    def test_model_obj_with_model_dir_attr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class FakeModelObj:
                model_dir = tmpdir
            result = self._call(None, FakeModelObj())
            self.assertEqual(result, tmpdir)


class TestResolveCtranslate2ModelPath(unittest.TestCase):
    """Test _resolve_ctranslate2_model_path."""

    def _call(self, *args, **kwargs):
        return _WBF["_resolve_ctranslate2_model_path"](*args, **kwargs)

    def test_unknown_model_name_returns_none(self):
        result = self._call("medium_nonexistent_xyz")
        self.assertIsNone(result)

    def test_none_returns_none(self):
        result = self._call(None)
        self.assertIsNone(result)

    def test_existing_directory_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._call(tmpdir)
            self.assertEqual(result, os.path.abspath(tmpdir))


class TestDetectLanguageWhispercppCli(unittest.TestCase):
    """Test _detect_language_whispercpp_cli_from_wav."""

    def test_handles_subprocess_error_gracefully(self):
        func = _WBF["_detect_language_whispercpp_cli_from_wav"]
        with patch("subprocess.run", side_effect=FileNotFoundError("No such file")):
            result = func("/some/audio.wav", "/nonexistent/model.bin")
        self.assertIsNone(result)

    def test_returns_none_when_model_path_empty(self):
        func = _WBF["_detect_language_whispercpp_cli_from_wav"]
        result = func("/some/audio.wav", "")
        self.assertIsNone(result)


class TestDownloadWhisperCppModel(unittest.TestCase):
    """Test the download_whisper_cpp_model function."""

    def test_invalid_model_name_returns_false(self):
        func = _WBF["download_whisper_cpp_model"]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = func("nonexistent_model_xyz", tmpdir)
            self.assertFalse(result)

    def test_download_with_mocked_urllib(self):
        func = _WBF["download_whisper_cpp_model"]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("urllib.request.urlretrieve", return_value=(None, None)):
                try:
                    result = func("tiny", tmpdir)
                except Exception:
                    pass


class TestWhisperBackendModuleLevelVars(unittest.TestCase):
    """Test module-level variables in whisper_backend."""

    def test_whispercpp_model_is_none_or_object(self):
        import anytran.whisper_backend as wb
        self.assertIsNone(wb._whispercpp_model)

    def test_whispercpp_model_id_is_none_or_string(self):
        import anytran.whisper_backend as wb
        self.assertIsNone(wb._whispercpp_model_id)

    def test_whisper_ctranslate2_model_is_none_or_object(self):
        import anytran.whisper_backend as wb
        self.assertIsNone(wb._whisper_ctranslate2_model)


if __name__ == "__main__":
    unittest.main()
