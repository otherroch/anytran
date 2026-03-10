"""Tests for --config and --genconfig CLI options."""
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# ---------------------------------------------------------------------------
# Minimal stubs so main.py can be imported without heavy dependencies
# ---------------------------------------------------------------------------
_EXT_STUBS = [
    "soundfile", "librosa", "paho", "paho.mqtt", "paho.mqtt.client",
    "faster_whisper", "gtts", "playsound3", "piper", "silero_vad",
    "moviepy", "moviepy.editor", "pydub", "av",
]
for _mod in _EXT_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
if "torch.cuda" not in sys.modules:
    sys.modules["torch.cuda"] = types.ModuleType("torch.cuda")
_torch = sys.modules["torch"]
_torch_cuda = sys.modules["torch.cuda"]
_torch_cuda.is_available = lambda: False
_torch.cuda = _torch_cuda
_torch.no_grad = lambda: None
_torch.inference_mode = lambda: None

if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.ModuleType("transformers")

if "googletrans" not in sys.modules:
    sys.modules["googletrans"] = types.ModuleType("googletrans")
    sys.modules["googletrans"].Translator = object

for _mod in (
    "anytran.tts", "anytran.mqtt_client", "anytran.vad",
    "anytran.whisper_backend", "anytran.audio_io",
    "anytran.stream_output", "anytran.stream_rtsp",
    "anytran.stream_youtube", "anytran.chatlog", "anytran.voice_matcher",
    "anytran.runners", "anytran.runners.run_file_input",
    "anytran.runners.run_multi_rtsp", "anytran.runners.run_realtime_output",
    "anytran.runners.run_realtime_rtsp", "anytran.runners.run_realtime_youtube",
    "anytran.processing", "anytran.text_translator",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Attributes needed by main imports
_wb = sys.modules["anytran.whisper_backend"]
if not hasattr(_wb, "download_whisper_cpp_model"):
    _wb.download_whisper_cpp_model = lambda *a, **kw: None
if not hasattr(_wb, "_derive_whispercpp_model_name"):
    _wb._derive_whispercpp_model_name = lambda *a, **kw: ""

_so = sys.modules["anytran.stream_output"]
if not hasattr(_so, "list_wasapi_loopback_devices"):
    _so.list_wasapi_loopback_devices = lambda: None

_vad = sys.modules["anytran.vad"]
if not hasattr(_vad, "SILERO_AVAILABLE"):
    _vad.SILERO_AVAILABLE = False

_vm = sys.modules["anytran.voice_matcher"]

_tt = sys.modules["anytran.text_translator"]
for _fn in ("set_translation_backend", "set_libretranslate_config",
            "set_translategemma_config",
            "set_metanllb_config", "set_marianmt_config"):
    if not hasattr(_tt, _fn):
        setattr(_tt, _fn, lambda *a, **kw: None)

_runners = sys.modules["anytran.runners"]
for _fn in ("run_file_input", "run_multi_rtsp", "run_realtime_output",
            "run_realtime_rtsp", "run_realtime_youtube"):
    if not hasattr(_runners, _fn):
        setattr(_runners, _fn, lambda *a, **kw: None)

# Web server stub
if "anytran.web_server" not in sys.modules:
    sys.modules["anytran.web_server"] = types.ModuleType("anytran.web_server")
if not hasattr(sys.modules["anytran.web_server"], "run_web_server"):
    sys.modules["anytran.web_server"].run_web_server = lambda *a, **kw: None

# Certs stub
if "anytran.certs" not in sys.modules:
    sys.modules["anytran.certs"] = types.ModuleType("anytran.certs")
if not hasattr(sys.modules["anytran.certs"], "generate_self_signed_cert"):
    sys.modules["anytran.certs"].generate_self_signed_cert = lambda *a, **kw: None

# Do NOT stub anytran.normalizer — it only uses stdlib and must be imported
# as the real module so that other test files that also import it are not broken.

# Utils stub
if "anytran.utils" not in sys.modules:
    sys.modules["anytran.utils"] = types.ModuleType("anytran.utils")
if not hasattr(sys.modules["anytran.utils"], "resolve_path_with_fallback"):
    sys.modules["anytran.utils"].resolve_path_with_fallback = lambda p, *a: p

# Config stub (the real one is simple, but stub to be safe)
for _key in list(sys.modules.keys()):
    if _key == "anytran.config":
        del sys.modules[_key]

# ---------------------------------------------------------------------------
# Now import the functions we want to test
# ---------------------------------------------------------------------------
from anytran.config_options import (  # noqa: E402
    _detect_format,
    _dict_to_toml,
    _get_default_config,
    _load_config_file,
    _write_config_file,
)
from anytran.main import main  # noqa: E402


class TestGetDefaultConfig(unittest.TestCase):
    """Tests for _get_default_config()."""

    def test_returns_dict(self):
        cfg = _get_default_config()
        self.assertIsInstance(cfg, dict)

    def test_contains_expected_keys(self):
        cfg = _get_default_config()
        for key in ("output_lang", "scribe_backend", "slate_backend",
                    "voice_backend", "verbose", "window_seconds"):
            self.assertIn(key, cfg)

    def test_default_output_lang(self):
        cfg = _get_default_config()
        self.assertEqual(cfg["output_lang"], "en")

    def test_default_scribe_backend(self):
        cfg = _get_default_config()
        self.assertEqual(cfg["scribe_backend"], "faster-whisper")

    def test_boolean_defaults_are_false(self):
        cfg = _get_default_config()
        for key in ("verbose", "timers", "timers_all", "lang_prefix",
                    "keep_temp", "dedup", "no_norm", "no_input_norm",
                    "scribe_vad", "no_auto_download", "web_ssl_self_signed"):
            self.assertFalse(cfg[key], f"Expected {key} to default to False")

    def test_all_values_are_json_serialisable(self):
        cfg = _get_default_config()
        # Should not raise
        json.dumps(cfg)


class TestLoadConfigFile(unittest.TestCase):
    """Tests for _load_config_file()."""

    def test_loads_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cfg.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"output_lang": "fr", "verbose": True}, fh)
            result = _load_config_file(path)
        self.assertEqual(result["output_lang"], "fr")
        self.assertTrue(result["verbose"])

    def test_exits_on_missing_file(self):
        with self.assertRaises(SystemExit):
            _load_config_file("/nonexistent/path/config.json")

    def test_exits_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("not valid json {{{")
            with self.assertRaises(SystemExit):
                _load_config_file(path)

    def test_exits_on_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "list.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([1, 2, 3], fh)
            with self.assertRaises(SystemExit):
                _load_config_file(path)


class TestWriteConfigFile(unittest.TestCase):
    """Tests for _write_config_file()."""

    def test_writes_to_file(self):
        cfg = {"output_lang": "es", "verbose": False}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.json")
            _write_config_file(cfg, path)
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        self.assertEqual(data["output_lang"], "es")

    def test_prints_to_stdout_when_dash(self):
        cfg = {"output_lang": "de"}
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            _write_config_file(cfg, "-")
        output = captured.getvalue()
        data = json.loads(output)
        self.assertEqual(data["output_lang"], "de")

    def test_prints_to_stdout_when_none(self):
        cfg = {"output_lang": "ja"}
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            _write_config_file(cfg, None)
        output = captured.getvalue()
        data = json.loads(output)
        self.assertEqual(data["output_lang"], "ja")


# ---------------------------------------------------------------------------
# TOML-specific tests
# ---------------------------------------------------------------------------

class TestDetectFormat(unittest.TestCase):
    """Tests for _detect_format()."""

    def test_toml_extension(self):
        self.assertEqual(_detect_format("config.toml"), "toml")

    def test_toml_extension_uppercase(self):
        self.assertEqual(_detect_format("config.TOML"), "toml")

    def test_json_extension(self):
        self.assertEqual(_detect_format("config.json"), "json")

    def test_no_extension_defaults_to_json(self):
        self.assertEqual(_detect_format("config"), "json")

    def test_none_defaults_to_json(self):
        self.assertEqual(_detect_format(None), "json")


class TestDictToToml(unittest.TestCase):
    """Tests for _dict_to_toml()."""

    def test_string_value(self):
        result = _dict_to_toml({"output_lang": "fr"})
        self.assertIn('output_lang = "fr"', result)

    def test_int_value(self):
        result = _dict_to_toml({"mqtt_port": 1883})
        self.assertIn("mqtt_port = 1883", result)

    def test_float_value(self):
        result = _dict_to_toml({"window_seconds": 5.0})
        self.assertIn("window_seconds = 5.0", result)

    def test_bool_true(self):
        result = _dict_to_toml({"verbose": True})
        self.assertIn("verbose = true", result)

    def test_bool_false(self):
        result = _dict_to_toml({"verbose": False})
        self.assertIn("verbose = false", result)

    def test_none_value_omitted(self):
        result = _dict_to_toml({"input": None, "output_lang": "en"})
        self.assertNotIn("input", result)
        self.assertIn('output_lang = "en"', result)

    def test_string_escaping(self):
        result = _dict_to_toml({"path": 'C:\\Users\\"test"'})
        # backslash → \\, quote → \" — both must be present
        self.assertIn('path = "C:\\\\Users\\\\\\"test\\""', result)

    def test_list_string_escaping(self):
        result = _dict_to_toml({"items": ['a"b', 'c\\d']})
        self.assertIn('items = ["a\\"b", "c\\\\d"]', result)


class TestLoadConfigFileToml(unittest.TestCase):
    """TOML-specific tests for _load_config_file()."""

    def test_loads_valid_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cfg.toml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('output_lang = "de"\nverbose = true\n')
            result = _load_config_file(path)
        self.assertEqual(result["output_lang"], "de")
        self.assertTrue(result["verbose"])

    def test_exits_on_invalid_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.toml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("this is not valid toml ===\n")
            with self.assertRaises(SystemExit):
                _load_config_file(path)

    def test_exits_on_missing_toml_file(self):
        with self.assertRaises(SystemExit):
            _load_config_file("/nonexistent/path/config.toml")

    def test_roundtrip_toml(self):
        """_write_config_file → _load_config_file round-trip for TOML."""
        cfg = {"output_lang": "es", "verbose": False, "mqtt_port": 1883,
               "window_seconds": 5.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.toml")
            _write_config_file(cfg, path)
            loaded = _load_config_file(path)
        for key, value in cfg.items():
            self.assertEqual(loaded[key], value)


class TestWriteConfigFileToml(unittest.TestCase):
    """TOML-specific tests for _write_config_file()."""

    def test_writes_toml_file(self):
        cfg = {"output_lang": "it", "verbose": False, "mqtt_port": 1883}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.toml")
            _write_config_file(cfg, path)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        self.assertIn('output_lang = "it"', content)
        self.assertIn("verbose = false", content)
        self.assertIn("mqtt_port = 1883", content)

    def test_toml_none_values_omitted(self):
        cfg = {"input": None, "output_lang": "fr"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.toml")
            _write_config_file(cfg, path)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        self.assertNotIn("input", content)
        self.assertIn('output_lang = "fr"', content)


class TestGenconfigToml(unittest.TestCase):
    """Tests for --genconfig with a .toml path."""

    def test_genconfig_writes_toml_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "defaults.toml")
            with patch("sys.argv", ["anytran", "--genconfig", out_path]):
                result = main()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as fh:
                content = fh.read()
        # At minimum, output_lang should be present
        self.assertIn('output_lang = "en"', content)


class TestConfigToml(unittest.TestCase):
    """Integration: --config with a .toml file."""

    def test_config_loads_toml(self):
        """Settings from a TOML config file are applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_path = os.path.join(tmpdir, "out.txt")
            with open(input_path, "w", encoding="utf-8") as fh:
                fh.write("hello")
            cfg_path = os.path.join(tmpdir, "test.toml")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write(
                    f'input = "{input_path.replace("\\", "\\\\")}"\n'
                    f'input_lang = "en"\n'
                    f'output_lang = "pt"\n'
                    f'slate_text = "{slate_path.replace("\\", "\\\\")}"\n'
                    f'slate_backend = "none"\n'
                )

            captured_args = {}

            def fake_run_file_input(path, input_lang, output_lang, *args, **kwargs):
                captured_args["output_lang"] = output_lang

            with patch("sys.argv", ["anytran", "--config", cfg_path]):
                with patch("anytran.pipelines.run_file_input", fake_run_file_input):
                    with patch("anytran.main.set_translation_backend"):
                        main()

        self.assertEqual(captured_args.get("output_lang"), "pt")

    def test_cli_arg_overrides_toml_config(self):
        """An explicit CLI --output-lang overrides the value from a TOML config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_path = os.path.join(tmpdir, "out.txt")
            with open(input_path, "w", encoding="utf-8") as fh:
                fh.write("hello")
            cfg_path = os.path.join(tmpdir, "test.toml")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write(
                    f'input = "{input_path.replace("\\", "\\\\")}"\n'
                    f'input_lang = "en"\n'
                    f'output_lang = "pt"\n'
                    f'slate_text = "{slate_path.replace("\\", "\\\\")}"\n'
                    f'slate_backend = "none"\n'
                )

            captured_args = {}

            def fake_run_file_input(path, input_lang, output_lang, *args, **kwargs):
                captured_args["output_lang"] = output_lang

            with patch("sys.argv", [
                "anytran", "--config", cfg_path, "--output-lang", "ja",
            ]):
                with patch("anytran.pipelines.run_file_input", fake_run_file_input):
                    with patch("anytran.main.set_translation_backend"):
                        main()

        self.assertEqual(captured_args.get("output_lang"), "ja")


class TestGenconfigOption(unittest.TestCase):
    """Tests for the --genconfig CLI option."""

    def test_genconfig_writes_file_and_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "myconfig.json")
            with patch("sys.argv", ["anytran", "--genconfig", out_path]):
                result = main()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["output_lang"], "en")
    def test_genconfig_stdout(self):
        import io
        import contextlib
        with patch("sys.argv", ["anytran", "--genconfig", "-"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = main()
            output = buf.getvalue()
        data = json.loads(output)
        self.assertEqual(result, 0)
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("output_lang"), "en")

    def test_genconfig_does_not_require_input_source(self):
        """--genconfig exits before the input-source group is validated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "cfg.json")
            # No --input / --web / --rtsp etc. provided
            with patch("sys.argv", ["anytran", "--genconfig", out_path]):
                result = main()
        self.assertEqual(result, 0)

    def test_genconfig_captures_cli_values(self):
        """--genconfig must write the actual CLI-supplied values, not static defaults.

        Regression test for the bug where three different command lines all
        produced identical output because genconfig was handled in the pre-parse
        step before the main parser ran.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "actual.json")
            with patch("sys.argv", [
                "anytran",
                "--output-lang", "fr",
                "--timers",
                "--keep-temp",
                "--slate-backend", "marianmt",
                "--genconfig", out_path,
            ]):
                result = main()
            self.assertEqual(result, 0)
            with open(out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        self.assertEqual(data["output_lang"], "fr")
        self.assertTrue(data["timers"])
        self.assertTrue(data["keep_temp"])
        self.assertEqual(data["slate_backend"], "marianmt")

    def test_genconfig_different_flags_produce_different_files(self):
        """Different CLI flag combinations must produce different config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_marianmt = os.path.join(tmpdir, "news.json")
            path_libretranslate = os.path.join(tmpdir, "news1.json")
            path_defaults = os.path.join(tmpdir, "news2.json")

            with patch("sys.argv", [
                "anytran", "--timers", "--keep-temp",
                "--output-lang", "fr", "--slate-backend", "marianmt",
                "--genconfig", path_marianmt,
            ]):
                main()

            with patch("sys.argv", [
                "anytran", "--timers", "--keep-temp",
                "--output-lang", "fr", "--slate-backend", "libretranslate",
                "--libretranslate-url", "http://127.0.0.1:5001",
                "--genconfig", path_libretranslate,
            ]):
                main()

            with patch("sys.argv", [
                "anytran", "--timers", "--output-lang", "fr",
                "--genconfig", path_defaults,
            ]):
                main()

            with open(path_marianmt, encoding="utf-8") as fh:
                data_marianmt = json.load(fh)
            with open(path_libretranslate, encoding="utf-8") as fh:
                data_libretranslate = json.load(fh)
            with open(path_defaults, encoding="utf-8") as fh:
                data_defaults = json.load(fh)

        # slate_backend must differ
        self.assertEqual(data_marianmt["slate_backend"], "marianmt")
        self.assertEqual(data_libretranslate["slate_backend"], "libretranslate")
        self.assertEqual(data_defaults["slate_backend"], "googletrans")
        # libretranslate_url must differ
        self.assertEqual(data_libretranslate["libretranslate_url"], "http://127.0.0.1:5001")
        self.assertIsNone(data_marianmt["libretranslate_url"])
        # keep_temp must differ between first two and the third
        self.assertTrue(data_marianmt["keep_temp"])
        self.assertTrue(data_libretranslate["keep_temp"])
        self.assertFalse(data_defaults["keep_temp"])
    def test_genconfig_prints_non_defaults(self):
        """--genconfig prints non-default settings to stdout before writing the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "cfg.json")
            captured = io.StringIO()
            with patch("sys.argv", [
                "anytran",
                "--output-lang", "fr",
                "--slate-backend", "marianmt",
                "--genconfig", out_path,
            ]):
                with patch("sys.stdout", captured):
                    main()
        output = captured.getvalue()
        self.assertIn("--output-lang", output)
        self.assertIn("fr", output)
        self.assertIn("--slate-backend", output)
        self.assertIn("marianmt", output)

    def test_genconfig_with_config_and_cli_override_prints_non_defaults(self):
        """--config + --genconfig: non-default output reflects config file AND CLI overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "result.json")
            cfg_path = os.path.join(tmpdir, "base.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump({"output_lang": "de", "slate_backend": "googletrans"}, fh)
            captured = io.StringIO()
            with patch("sys.argv", [
                "anytran",
                "--config", cfg_path,
                "--slate-backend", "marianmt",
                "--genconfig", out_path,
            ]):
                with patch("sys.stdout", captured):
                    main()
            output = captured.getvalue()
            # --output-lang de came from the config file, so it's non-default
            self.assertIn("--output-lang", output)
            self.assertIn("de", output)
            # --slate-backend marianmt was overridden on the CLI
            self.assertIn("--slate-backend", output)
            self.assertIn("marianmt", output)
            # The written file should reflect CLI override
            with open(out_path, encoding="utf-8") as fh:
                data = json.load(fh)
        self.assertEqual(data["output_lang"], "de")
        self.assertEqual(data["slate_backend"], "marianmt")


class TestConfigOption(unittest.TestCase):
    """Tests for the --config CLI option (config file values used as defaults,
    overridden by explicit CLI args)."""

    def _make_config(self, tmpdir, overrides=None):
        cfg = _get_default_config()
        if overrides:
            cfg.update(overrides)
        path = os.path.join(tmpdir, "test_config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        return path

    def test_config_sets_output_lang(self):
        """Config file value for output_lang is applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_path = os.path.join(tmpdir, "out.txt")
            with open(input_path, "w", encoding="utf-8") as fh:
                fh.write("hello")
            cfg_path = self._make_config(tmpdir, {
                "input": input_path,
                "input_lang": "en",
                "output_lang": "fr",
                "slate_text": slate_path,
                "slate_backend": "none",
            })

            captured_args = {}

            def fake_run_file_input(path, input_lang, output_lang, *args, **kwargs):
                captured_args["output_lang"] = output_lang

            with patch("sys.argv", ["anytran", "--config", cfg_path]):
                with patch("anytran.pipelines.run_file_input", fake_run_file_input):
                    with patch("anytran.main.set_translation_backend"):
                        main()

            self.assertEqual(captured_args.get("output_lang"), "fr")

    def test_cli_arg_overrides_config(self):
        """An explicit CLI --output-lang overrides the value from the config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_path = os.path.join(tmpdir, "out.txt")
            with open(input_path, "w", encoding="utf-8") as fh:
                fh.write("hello")
            # Config says fr, CLI will say de
            cfg_path = self._make_config(tmpdir, {
                "input": input_path,
                "input_lang": "en",
                "output_lang": "fr",
                "slate_text": slate_path,
                "slate_backend": "none",
            })

            captured_args = {}

            def fake_run_file_input(path, input_lang, output_lang, *args, **kwargs):
                captured_args["output_lang"] = output_lang

            with patch("sys.argv", [
                "anytran", "--config", cfg_path, "--output-lang", "de",
            ]):
                with patch("anytran.pipelines.run_file_input", fake_run_file_input):
                    with patch("anytran.main.set_translation_backend"):
                        main()

            self.assertEqual(captured_args.get("output_lang"), "de")

    def test_config_missing_file_exits(self):
        """--config with a nonexistent file exits with an error."""
        with patch("sys.argv", [
            "anytran", "--config", "/no/such/file.json",
            "--input", "dummy.txt", "--input-lang", "en",
        ]):
            with self.assertRaises(SystemExit):
                main()


class TestBooleanOptionInversion(unittest.TestCase):
    """Tests for inverted boolean CLI options (e.g. --norm inverts --no-norm)."""

    def _parse_genconfig(self, argv):
        """Helper: run main() with --genconfig and return the parsed config dict."""
        captured = io.StringIO()
        with patch("sys.argv", argv + ["--genconfig", "-"]):
            with patch("sys.stdout", captured):
                result = main()
        self.assertEqual(result, 0)
        output = captured.getvalue()
        # The JSON starts after any "Non-default settings:" header
        json_start = output.find("{")
        return json.loads(output[json_start:])

    # --no-norm / --norm
    def test_no_norm_sets_true(self):
        data = self._parse_genconfig(["anytran", "--no-norm"])
        self.assertTrue(data["no_norm"])

    def test_norm_inverts_no_norm(self):
        data = self._parse_genconfig(["anytran", "--norm"])
        self.assertFalse(data["no_norm"])

    def test_norm_overrides_no_norm(self):
        """Last flag wins: --no-norm then --norm should result in no_norm=False."""
        data = self._parse_genconfig(["anytran", "--no-norm", "--norm"])
        self.assertFalse(data["no_norm"])

    # --no-input-norm / --input-norm
    def test_no_input_norm_sets_true(self):
        data = self._parse_genconfig(["anytran", "--no-input-norm"])
        self.assertTrue(data["no_input_norm"])

    def test_input_norm_inverts_no_input_norm(self):
        data = self._parse_genconfig(["anytran", "--input-norm"])
        self.assertFalse(data["no_input_norm"])

    # --timers / --no-timers
    def test_timers_sets_true(self):
        data = self._parse_genconfig(["anytran", "--timers"])
        self.assertTrue(data["timers"])

    def test_no_timers_inverts_timers(self):
        data = self._parse_genconfig(["anytran", "--no-timers"])
        self.assertFalse(data["timers"])

    def test_no_timers_overrides_timers(self):
        """Last flag wins: --timers then --no-timers should result in timers=False."""
        data = self._parse_genconfig(["anytran", "--timers", "--no-timers"])
        self.assertFalse(data["timers"])

    # --keep-temp / --no-keep-temp
    def test_keep_temp_sets_true(self):
        data = self._parse_genconfig(["anytran", "--keep-temp"])
        self.assertTrue(data["keep_temp"])

    def test_no_keep_temp_inverts_keep_temp(self):
        data = self._parse_genconfig(["anytran", "--no-keep-temp"])
        self.assertFalse(data["keep_temp"])

    # --verbose / --no-verbose
    def test_verbose_sets_true(self):
        data = self._parse_genconfig(["anytran", "--verbose"])
        self.assertTrue(data["verbose"])

    def test_no_verbose_inverts_verbose(self):
        data = self._parse_genconfig(["anytran", "--no-verbose"])
        self.assertFalse(data["verbose"])

    # --dedup / --no-dedup
    def test_dedup_sets_true(self):
        data = self._parse_genconfig(["anytran", "--dedup"])
        self.assertTrue(data["dedup"])

    def test_no_dedup_inverts_dedup(self):
        data = self._parse_genconfig(["anytran", "--no-dedup"])
        self.assertFalse(data["dedup"])

    # --lang-prefix / --no-lang-prefix
    def test_lang_prefix_sets_true(self):
        data = self._parse_genconfig(["anytran", "--lang-prefix"])
        self.assertTrue(data["lang_prefix"])

    def test_no_lang_prefix_inverts_lang_prefix(self):
        data = self._parse_genconfig(["anytran", "--no-lang-prefix"])
        self.assertFalse(data["lang_prefix"])

    # --timers-all / --no-timers-all
    def test_timers_all_sets_true(self):
        data = self._parse_genconfig(["anytran", "--timers-all"])
        self.assertTrue(data["timers_all"])

    def test_no_timers_all_inverts_timers_all(self):
        data = self._parse_genconfig(["anytran", "--no-timers-all"])
        self.assertFalse(data["timers_all"])

    # --no-auto-download / --auto-download
    def test_no_auto_download_sets_true(self):
        data = self._parse_genconfig(["anytran", "--no-auto-download"])
        self.assertTrue(data["no_auto_download"])

    def test_auto_download_inverts_no_auto_download(self):
        data = self._parse_genconfig(["anytran", "--auto-download"])
        self.assertFalse(data["no_auto_download"])


class TestVoiceBackendCLI(unittest.TestCase):
    """Tests for --voice-backend coqui CLI option."""

    def _parse_genconfig(self, argv):
        """Helper: run main() with --genconfig and return the parsed config dict."""
        captured = io.StringIO()
        with patch("sys.argv", argv + ["--genconfig", "-"]):
            with patch("sys.stdout", captured):
                result = main()
        self.assertEqual(result, 0)
        output = captured.getvalue()
        json_start = output.find("{")
        return json.loads(output[json_start:])

    def test_voice_backend_coqui_accepted(self):
        """--voice-backend coqui must be accepted without error."""
        data = self._parse_genconfig(["anytran", "--voice-backend", "coqui"])
        self.assertEqual(data["voice_backend"], "coqui")

    def test_voice_backend_coqui_with_voice_match(self):
        """--voice-backend coqui --voice-match must be accepted."""
        data = self._parse_genconfig([
            "anytran", "--voice-backend", "coqui", "--voice-match",
        ])
        self.assertEqual(data["voice_backend"], "coqui")
        self.assertTrue(data["voice_match"])

    def test_voice_backend_coqui_with_voice_model(self):
        """--voice-backend coqui accepts a custom model via --voice-model."""
        model = "tts_models/multilingual/multi-dataset/xtts_v2"
        data = self._parse_genconfig([
            "anytran", "--voice-backend", "coqui", "--voice-model", model,
        ])
        self.assertEqual(data["voice_backend"], "coqui")
        self.assertEqual(data["voice_model"], model)

    def test_voice_backend_invalid_choice_rejected(self):
        """An unrecognised --voice-backend value must cause SystemExit."""
        with self.assertRaises(SystemExit):
            with patch("sys.argv", [
                "anytran", "--voice-backend", "notabackend", "--genconfig", "-",
            ]):
                main()

    def test_all_backend_choices_accepted(self):
        """Every documented --voice-backend choice must parse without error."""
        for backend in ("auto", "gtts", "piper", "custom", "fish", "indextts", "coqui"):
            data = self._parse_genconfig(["anytran", "--voice-backend", backend])
            self.assertEqual(data["voice_backend"], backend,
                             f"Expected voice_backend={backend!r}")


if __name__ == "__main__":
    unittest.main()
