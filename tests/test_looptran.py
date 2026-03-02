import os
import sys
import tempfile
import types
import unittest
from unittest.mock import call, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Stub out heavy optional dependencies so that importing anytran.pipelines does not
# require soundfile, librosa, torch, transformers, paho, etc. to be installed.
_STUB_MODULES = [
    "soundfile", "librosa", "paho", "paho.mqtt", "paho.mqtt.client",
    "torch", "transformers", "faster_whisper", "gtts", "playsound3",
    "libretranslate", "piper", "silero_vad",
    "moviepy", "moviepy.editor",
    "anytran.audio_io", "anytran.whisper_backend", "anytran.vad",
    "anytran.voice_matcher", "anytran.stream_output", "anytran.stream_youtube",
    "anytran.stream_rtsp", "anytran.chatlog", "anytran.web_server",
    "anytran.tts", "anytran.processing",
    "anytran.runners.run_realtime_rtsp", "anytran.runners.run_multi_rtsp",
    "anytran.runners.run_realtime_output", "anytran.runners.run_realtime_youtube",
    "anytran.runners.run_file_input",
    "uvicorn", "fastapi", "starlette",
]
for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# Provide the minimal attributes that anytran modules need during import
_audio_io = sys.modules["anytran.audio_io"]
_audio_io.load_audio_any = MagicMock()
_audio_io.output_audio = MagicMock()

_vad = sys.modules["anytran.vad"]
_vad.SILERO_AVAILABLE = False

_voice_matcher = sys.modules["anytran.voice_matcher"]

_stream_output = sys.modules["anytran.stream_output"]
_stream_output.list_wasapi_loopback_devices = MagicMock()
_stream_output.get_wasapi_loopback_device_info = MagicMock()
_stream_output.stream_output_audio = MagicMock()

_web_server = sys.modules["anytran.web_server"]
_web_server.run_web_server = MagicMock()

_run_file_mod = sys.modules["anytran.runners.run_file_input"]
_run_file_mod.run_file_input = MagicMock()

for _runner_mod in [
    "anytran.runners.run_realtime_rtsp",
    "anytran.runners.run_multi_rtsp",
    "anytran.runners.run_realtime_output",
    "anytran.runners.run_realtime_youtube",
]:
    _m = sys.modules[_runner_mod]
    fn_name = _runner_mod.split(".")[-1]
    setattr(_m, fn_name, MagicMock())

# Stub anytran.runners package itself
import importlib
if "anytran.runners" not in sys.modules:
    _runners_pkg = types.ModuleType("anytran.runners")
    sys.modules["anytran.runners"] = _runners_pkg
_runners = sys.modules["anytran.runners"]
_runners.run_file_input = MagicMock()
_runners.run_multi_rtsp = MagicMock()
_runners.run_realtime_output = MagicMock()
_runners.run_realtime_rtsp = MagicMock()
_runners.run_realtime_youtube = MagicMock()

# Stub text_translator
if "anytran.text_translator" not in sys.modules:
    _tt = types.ModuleType("anytran.text_translator")
    sys.modules["anytran.text_translator"] = _tt
_tt = sys.modules["anytran.text_translator"]
for _fn in [
    "set_translation_backend", "set_libretranslate_config",
    "set_translategemma_config", "set_metanllb_config", "set_marianmt_config",
    "translate_text",
]:
    setattr(_tt, _fn, MagicMock())

# Stub config
if "anytran.config" not in sys.modules:
    _cfg = types.ModuleType("anytran.config")
    sys.modules["anytran.config"] = _cfg
_cfg = sys.modules["anytran.config"]
for _fn in [
    "set_whisper_backend", "set_whisper_cpp_config", "set_whispercpp_cli_detect_lang",
    "set_whispercpp_force_cli", "set_whisper_ctranslate2_config",
]:
    setattr(_cfg, _fn, MagicMock())

# Stub whisper_backend
if "anytran.whisper_backend" not in sys.modules:
    _wb = types.ModuleType("anytran.whisper_backend")
    sys.modules["anytran.whisper_backend"] = _wb
_wb = sys.modules["anytran.whisper_backend"]
_wb.download_whisper_cpp_model = MagicMock()
_wb._derive_whispercpp_model_name = MagicMock()

# Stub utils
if "anytran.utils" not in sys.modules:
    _utils = types.ModuleType("anytran.utils")
    sys.modules["anytran.utils"] = _utils
_utils = sys.modules["anytran.utils"]
_utils.resolve_path_with_fallback = MagicMock()

# Stub certs
if "anytran.certs" not in sys.modules:
    _certs = types.ModuleType("anytran.certs")
    sys.modules["anytran.certs"] = _certs
_certs = sys.modules["anytran.certs"]
_certs.generate_self_signed_cert = MagicMock()

# Now that all heavy modules are stubbed, we can import pipelines safely
import anytran.pipelines as _cli_new


class _FakeArgs:
    """Minimal args object for testing _run_file_pipeline."""

    def __init__(self, input_path, input_lang, output_lang, slate_text, looptran=0, voice_lang=None, tran_converge=False):
        self.input = input_path
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.slate_text = slate_text
        self.looptran = looptran
        self.voice_lang = voice_lang
        # Map bool to the None/int convention used by argparse --tran-converge:
        # False -> None (disabled), True -> 0 (exact-match threshold)
        if tran_converge is False:
            self.tran_converge = None
        elif tran_converge is True:
            self.tran_converge = 0
        else:
            self.tran_converge = tran_converge
        self.batch_input_text = False


def _make_config(input_lang, output_lang, slate_text):
    needs_translation = output_lang.lower() != "en"
    return {
        "input_lang": input_lang,
        "output_lang": output_lang,
        "magnitude_threshold": 0.01,
        "scribe_voice": None,
        "slate_voice": None,
        "model": "tiny",
        "verbose": False,
        "mqtt_broker": None,
        "mqtt_port": 1883,
        "mqtt_username": None,
        "mqtt_password": None,
        "mqtt_topic": "translation",
        "scribe_vad": False,
        "voice_backend": "gtts",
        "voice_model": None,
        "window_seconds": 5.0,
        "overlap_seconds": 0.0,
        "timers": False,
        "timers_all": False,
        "scribe_backend": "auto",
        "text_translation_target": output_lang if needs_translation else None,
        "slate_backend": "googletrans",
        "voice_lang": output_lang if needs_translation else "en",
        "scribe_text": None,
        "slate_text": slate_text,
        "voice_match": False,
        "keep_temp": False,
        "dedup": False,
        "lang_prefix": False,
        "needs_translation": needs_translation,
    }


class TestLooptran(unittest.TestCase):
    """Tests for the --looptran option in _run_file_pipeline."""

    def _run_pipeline(self, input_path, input_lang, output_lang, slate_text, looptran, mock_rfi, tran_converge=False):
        """Helper: call _run_file_pipeline with mocked run_file_input."""
        args = _FakeArgs(input_path, input_lang, output_lang, slate_text, looptran=looptran, tran_converge=tran_converge)
        config = _make_config(input_lang, output_lang, slate_text)
        with patch.object(_cli_new, "run_file_input", mock_rfi):
            _cli_new._run_file_pipeline(args, config)

    def test_looptran_zero_calls_run_file_input_once(self):
        """With --looptran 0 (default), run_file_input is called exactly once."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=0, mock_rfi=mock_rfi)
        mock_rfi.assert_called_once()

    def test_looptran_one_calls_run_file_input_twice(self):
        """With --looptran 1, run_file_input is called twice (original + 1 loop)."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=1, mock_rfi=mock_rfi)
        self.assertEqual(mock_rfi.call_count, 2)

    def test_looptran_n_calls_run_file_input_n_plus_one_times(self):
        """With --looptran n, run_file_input is called n+1 times."""
        for n in [2, 3, 5]:
            mock_rfi = MagicMock()
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, "input.txt")
                slate_text = os.path.join(tmpdir, "slate.txt")
                open(input_path, "w").close()
                self._run_pipeline(input_path, "fr", "es", slate_text, looptran=n, mock_rfi=mock_rfi)
            self.assertEqual(mock_rfi.call_count, n + 1, f"Expected {n+1} calls for looptran={n}")

    def test_looptran_second_invocation_uses_slate_as_input(self):
        """Second invocation uses the slate-text file from the first as its input."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=1, mock_rfi=mock_rfi)

        # Second call's first positional arg should be the original slate_text
        second_call_args = mock_rfi.call_args_list[1]
        self.assertEqual(second_call_args[0][0], slate_text)

    def test_looptran_second_invocation_swaps_languages(self):
        """Second invocation swaps input-lang and output-lang."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=1, mock_rfi=mock_rfi)

        second_call_args = mock_rfi.call_args_list[1]
        # Positional args: (input_path, input_lang, output_lang, ...)
        self.assertEqual(second_call_args[0][1], "es")   # swapped input_lang
        self.assertEqual(second_call_args[0][2], "fr")   # swapped output_lang

    def test_looptran_second_invocation_creates_postfix_1_slate(self):
        """Second invocation creates a slate-text file with '_1' postfix."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=1, mock_rfi=mock_rfi)

        second_call_kwargs = mock_rfi.call_args_list[1][1]
        expected_new_slate = os.path.join(os.path.dirname(slate_text), "slate_1.txt")
        self.assertEqual(second_call_kwargs["slate_text_file"], expected_new_slate)

    def test_looptran_third_invocation_uses_postfix_1_as_input(self):
        """Third invocation (looptran=2) uses the '_1' slate file as its input."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=2, mock_rfi=mock_rfi)

        third_call_args = mock_rfi.call_args_list[2][0]
        expected_input = os.path.join(os.path.dirname(slate_text), "slate_1.txt")
        self.assertEqual(third_call_args[0], expected_input)

    def test_looptran_third_invocation_creates_postfix_2_slate(self):
        """Third invocation (looptran=2) creates a slate-text file with '_2' postfix."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=2, mock_rfi=mock_rfi)

        third_call_kwargs = mock_rfi.call_args_list[2][1]
        expected_new_slate = os.path.join(os.path.dirname(slate_text), "slate_2.txt")
        self.assertEqual(third_call_kwargs["slate_text_file"], expected_new_slate)

    def test_looptran_skipped_for_audio_input(self):
        """looptran is skipped when input file is not a text file."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", slate_text, looptran=2, mock_rfi=mock_rfi)
        # Only one call - the original, looptran does not apply to audio
        mock_rfi.assert_called_once()

    def test_looptran_skipped_when_same_lang(self):
        """looptran is skipped when input-lang and output-lang are the same."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "en", "en", slate_text, looptran=2, mock_rfi=mock_rfi)
        mock_rfi.assert_called_once()

    def test_looptran_skipped_when_no_slate_text(self):
        """looptran is skipped when --slate-text is not specified."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            open(input_path, "w").close()
            self._run_pipeline(input_path, "fr", "es", None, looptran=2, mock_rfi=mock_rfi)
        mock_rfi.assert_called_once()


class TestTranConverge(unittest.TestCase):
    """Tests for the --tran-converge option in _run_file_pipeline."""

    def _run_pipeline_converge(self, input_path, input_lang, output_lang, slate_text, looptran, mock_rfi,
                                slate_file_contents=None):
        """Helper: run pipeline with --tran-converge, writing file contents after each mock call."""
        args = _FakeArgs(input_path, input_lang, output_lang, slate_text, looptran=looptran, tran_converge=True)
        config = _make_config(input_lang, output_lang, slate_text)

        call_count = [0]

        def side_effect(*a, **kw):
            idx = call_count[0]
            call_count[0] += 1
            # Write the slate file that this call would have produced
            slate_path = kw.get("slate_text_file")
            if slate_path and slate_file_contents and idx < len(slate_file_contents):
                with open(slate_path, "w") as f:
                    f.write(slate_file_contents[idx])

        mock_rfi.side_effect = side_effect

        with patch.object(_cli_new, "run_file_input", mock_rfi):
            _cli_new._run_file_pipeline(args, config)

    def test_tran_converge_no_early_stop_when_files_differ(self):
        """With --tran-converge and no convergence, all iterations run."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            # All 5 calls produce different pairings, no convergence:
            # slate.txt="A", slate_1="B", slate_2="C", slate_3="D", slate_4="E"
            # Pairs checked: (slate_2 vs slate.txt)="C"vs"A", (slate_3 vs slate_1)="D"vs"B", (slate_4 vs slate_2)="E"vs"C"
            self._run_pipeline_converge(
                input_path, "fr", "es", slate_text, looptran=4, mock_rfi=mock_rfi,
                slate_file_contents=["A", "B", "C", "D", "E"],
            )
        # All 4 loop iterations run (plus the initial call = 5 total)
        self.assertEqual(mock_rfi.call_count, 5)

    def test_tran_converge_stops_early_when_files_match(self):
        """With --tran-converge, stops early when output matches the file two iterations back."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            # slate.txt="A", slate_1="B", slate_2="A"  -> convergence at iteration 2
            # (slate_2 matches slate.txt, both "A")
            self._run_pipeline_converge(
                input_path, "fr", "es", slate_text, looptran=6, mock_rfi=mock_rfi,
                slate_file_contents=["A", "B", "A"],
            )
        # Stopped after iteration 2: initial call + 2 loop iterations = 3 total
        self.assertEqual(mock_rfi.call_count, 3)

    def test_tran_converge_stops_at_correct_iteration(self):
        """Convergence check compares iteration i with iteration i-2 (same language)."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            # slate.txt="A", slate_1="B", slate_2="C", slate_3="D", slate_4="C"
            # Checks: (slate_2="C" vs slate.txt="A")=differ, (slate_3="D" vs slate_1="B")=differ,
            #         (slate_4="C" vs slate_2="C")=match -> stop at iteration 4
            self._run_pipeline_converge(
                input_path, "fr", "es", slate_text, looptran=6, mock_rfi=mock_rfi,
                slate_file_contents=["A", "B", "C", "D", "C"],
            )
        # Stopped after iteration 4: initial + 4 loop = 5 calls
        self.assertEqual(mock_rfi.call_count, 5)

    def test_tran_converge_without_flag_does_not_stop_early(self):
        """Without --tran-converge, matching files do not cause early stop."""
        mock_rfi = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_text = os.path.join(tmpdir, "slate.txt")
            open(input_path, "w").close()
            args = _FakeArgs(input_path, "fr", "es", slate_text, looptran=4, tran_converge=False)
            config = _make_config("fr", "es", slate_text)

            call_count = [0]
            contents = ["A", "B", "A", "B", "A"]

            def side_effect(*a, **kw):
                idx = call_count[0]
                call_count[0] += 1
                slate_path = kw.get("slate_text_file")
                if slate_path and idx < len(contents):
                    with open(slate_path, "w") as f:
                        f.write(contents[idx])

            mock_rfi.side_effect = side_effect
            with patch.object(_cli_new, "run_file_input", mock_rfi):
                _cli_new._run_file_pipeline(args, config)
        # All 4 loop iterations run despite matching files
        self.assertEqual(mock_rfi.call_count, 5)


class TestWebPipelineVoiceBackend(unittest.TestCase):
    """Ensure web pipeline passes voice backend settings to the server."""

    def test_web_pipeline_uses_configured_voice_backend(self):
        args = types.SimpleNamespace(
            web_host="0.0.0.0",
            web_port=8443,
            web_ssl_cert=None,
            web_ssl_key=None,
        )
        config = _make_config("auto", "test-lang", slate_text=None)
        config["voice_backend"] = "piper"
        config["voice_model"] = "fr_test_voice"

        with patch.object(_cli_new, "run_web_server", MagicMock()) as mock_run_web_server:
            _cli_new._run_web_pipeline(args, config)

        mock_run_web_server.assert_called_once()
        call_kwargs = mock_run_web_server.call_args.kwargs
        self.assertEqual(call_kwargs.get("voice_backend"), "piper")
        self.assertEqual(call_kwargs.get("voice_model"), "fr_test_voice")


if __name__ == "__main__":
    unittest.main()
