"""Tests for anytran.config module — covering setters and getters."""
import sys
import unittest

# ---------------------------------------------------------------------------
# Retrieve the real function references saved by conftest.py BEFORE
# test_looptran.py replaces them with MagicMock instances.
# ---------------------------------------------------------------------------
from tests.conftest import _real_config_funcs as _RF


def _set_whisper_backend(v): return _RF["set_whisper_backend"](v)
def _get_whisper_backend(): return _RF["get_whisper_backend"]()
def _set_whisper_cpp_config(**kw): return _RF["set_whisper_cpp_config"](**kw)
def _get_whisper_cpp_config(): return _RF["get_whisper_cpp_config"]()
def _set_whispercpp_cli_detect_lang(v): return _RF["set_whispercpp_cli_detect_lang"](v)
def _get_whispercpp_cli_detect_lang(): return _RF["get_whispercpp_cli_detect_lang"]()
def _set_whispercpp_force_cli(v): return _RF["set_whispercpp_force_cli"](v)
def _get_whispercpp_force_cli(): return _RF["get_whispercpp_force_cli"]()
def _set_whisper_ctranslate2_config(**kw): return _RF["set_whisper_ctranslate2_config"](**kw)
def _get_whisper_ctranslate2_config(): return _RF["get_whisper_ctranslate2_config"]()
def _parse_device_index(v): return _RF["_parse_whisper_ctranslate2_device_index"](v)


class TestWhisperBackendConfig(unittest.TestCase):
    def setUp(self):
        self._orig_backend = _get_whisper_backend()

    def tearDown(self):
        _set_whisper_backend(self._orig_backend)

    def test_set_and_get_whisper_backend(self):
        _set_whisper_backend("faster_whisper")
        self.assertEqual(_get_whisper_backend(), "faster_whisper")

    def test_set_whispercpp_cli_detect_lang_true(self):
        _set_whispercpp_cli_detect_lang(True)
        self.assertTrue(_get_whispercpp_cli_detect_lang())

    def test_set_whispercpp_cli_detect_lang_false(self):
        _set_whispercpp_cli_detect_lang(False)
        self.assertFalse(_get_whispercpp_cli_detect_lang())

    def test_set_whispercpp_force_cli(self):
        _set_whispercpp_force_cli(True)
        self.assertTrue(_get_whispercpp_force_cli())

    def test_set_whispercpp_force_cli_false(self):
        _set_whispercpp_force_cli(False)
        self.assertFalse(_get_whispercpp_force_cli())


class TestWhisperCppConfig(unittest.TestCase):
    def setUp(self):
        self._orig_config = dict(_get_whisper_cpp_config())

    def tearDown(self):
        _set_whisper_cpp_config(**self._orig_config)

    def test_set_bin_path(self):
        _set_whisper_cpp_config(bin_path="/usr/local/bin/whisper")
        self.assertEqual(_get_whisper_cpp_config()["bin_path"], "/usr/local/bin/whisper")

    def test_set_model_path(self):
        _set_whisper_cpp_config(model_path="/models/ggml-medium.bin")
        self.assertEqual(_get_whisper_cpp_config()["model_path"], "/models/ggml-medium.bin")

    def test_set_threads(self):
        _set_whisper_cpp_config(threads=4)
        self.assertEqual(_get_whisper_cpp_config()["threads"], 4)

    def test_none_values_not_overwritten(self):
        original = _get_whisper_cpp_config()
        _set_whisper_cpp_config(bin_path=None, model_path=None, threads=None)
        after = _get_whisper_cpp_config()
        self.assertEqual(original, after)

    def test_get_returns_copy(self):
        config1 = _get_whisper_cpp_config()
        config2 = _get_whisper_cpp_config()
        config1["bin_path"] = "modified"
        self.assertNotEqual(config1["bin_path"], config2["bin_path"])


class TestWhisperCtranslate2Config(unittest.TestCase):
    def setUp(self):
        self._orig_config = dict(_get_whisper_ctranslate2_config())

    def tearDown(self):
        _set_whisper_ctranslate2_config(**self._orig_config)

    def test_set_model_name(self):
        _set_whisper_ctranslate2_config(model_name="medium")
        self.assertEqual(_get_whisper_ctranslate2_config()["model_name"], "medium")

    def test_set_device(self):
        _set_whisper_ctranslate2_config(device="cpu")
        self.assertEqual(_get_whisper_ctranslate2_config()["device"], "cpu")

    def test_set_compute_type(self):
        _set_whisper_ctranslate2_config(compute_type="int8")
        self.assertEqual(_get_whisper_ctranslate2_config()["compute_type"], "int8")

    def test_set_device_index(self):
        _set_whisper_ctranslate2_config(device_index=1)
        self.assertEqual(_get_whisper_ctranslate2_config()["device_index"], 1)

    def test_none_values_not_overwritten(self):
        original = _get_whisper_ctranslate2_config()
        _set_whisper_ctranslate2_config(model_name=None, device=None)
        after = _get_whisper_ctranslate2_config()
        self.assertEqual(original, after)


class TestParseDeviceIndex(unittest.TestCase):
    """Test the internal device index parser."""

    def test_valid_integer_string(self):
        self.assertEqual(_parse_device_index("2"), 2)

    def test_none_returns_none(self):
        self.assertIsNone(_parse_device_index(None))

    def test_invalid_string_returns_none(self):
        result = _parse_device_index("not_a_number")
        self.assertIsNone(result)

    def test_zero_valid(self):
        self.assertEqual(_parse_device_index("0"), 0)


if __name__ == "__main__":
    unittest.main()
