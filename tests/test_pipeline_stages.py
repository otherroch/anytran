import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

# ============================================================================
# STUB SETUP — must happen before any anytran imports
# ============================================================================

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Stub heavy external packages that are not installed in the test environment.
_EXT_STUBS = [
    "soundfile", "librosa", "paho", "paho.mqtt", "paho.mqtt.client",
    "faster_whisper", "gtts", "playsound3", "piper", "silero_vad",
    "moviepy", "moviepy.editor", "pydub",
]
for _mod in _EXT_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Stub torch with attributes that text_translator.py needs at module level.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
if "torch.cuda" not in sys.modules:
    sys.modules["torch.cuda"] = types.ModuleType("torch.cuda")
_torch = sys.modules["torch"]
_torch_cuda = sys.modules["torch.cuda"]
_torch_cuda.is_available = MagicMock(return_value=False)
_torch.cuda = _torch_cuda
_torch.no_grad = MagicMock()
_torch.inference_mode = MagicMock()
for _attr in ("bfloat16", "float32", "float16"):
    if not hasattr(_torch, _attr):
        setattr(_torch, _attr, _attr)

# Stub transformers with every class imported by text_translator.py.
if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.ModuleType("transformers")
_transformers = sys.modules["transformers"]
for _cls in (
    "AutoProcessor", "AutoModelForImageTextToText", "pipeline",
    "AutoTokenizer", "M2M100ForConditionalGeneration",
    "AutoModelForSeq2SeqLM", "NllbTokenizer", "NllbTokenizerFast",
    "MarianMTModel", "MarianTokenizer",
):
    if not hasattr(_transformers, _cls):
        setattr(_transformers, _cls, MagicMock())

# Stub googletrans Translator used at module level in text_translator.py.
if "googletrans" not in sys.modules:
    sys.modules["googletrans"] = types.ModuleType("googletrans")
_googletrans = sys.modules["googletrans"]
if not hasattr(_googletrans, "Translator"):
    _googletrans.Translator = MagicMock()

# Ensure all anytran submodule stubs carry the attributes that processing.py,
# run_file_input.py, and the runners __init__.py require when they are imported.
for _mod in (
    "anytran.tts", "anytran.mqtt_client", "anytran.vad",
    "anytran.whisper_backend", "anytran.audio_io",
    "anytran.stream_output", "anytran.stream_rtsp",
    "anytran.chatlog", "anytran.voice_matcher",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

_anytran_tts = sys.modules["anytran.tts"]
for _fn in ("play_output", "synthesize_tts_pcm", "synthesize_tts_pcm_with_cloning"):
    if not hasattr(_anytran_tts, _fn):
        setattr(_anytran_tts, _fn, MagicMock())

_anytran_mqtt = sys.modules["anytran.mqtt_client"]
for _fn in ("init_mqtt", "send_mqtt_text"):
    if not hasattr(_anytran_mqtt, _fn):
        setattr(_anytran_mqtt, _fn, MagicMock())

_anytran_vad = sys.modules["anytran.vad"]
if not hasattr(_anytran_vad, "SILERO_AVAILABLE"):
    _anytran_vad.SILERO_AVAILABLE = False
if not hasattr(_anytran_vad, "has_speech_silero"):
    _anytran_vad.has_speech_silero = MagicMock(return_value=True)

_anytran_wb = sys.modules["anytran.whisper_backend"]
if not hasattr(_anytran_wb, "translate_audio"):
    _anytran_wb.translate_audio = MagicMock()

_anytran_audio = sys.modules["anytran.audio_io"]
for _fn in ("load_audio_any", "output_audio"):
    if not hasattr(_anytran_audio, _fn):
        setattr(_anytran_audio, _fn, MagicMock())

_anytran_so = sys.modules["anytran.stream_output"]
for _fn in ("get_wasapi_loopback_device_info", "stream_output_audio", "list_wasapi_loopback_devices"):
    if not hasattr(_anytran_so, _fn):
        setattr(_anytran_so, _fn, MagicMock())

_anytran_sr = sys.modules["anytran.stream_rtsp"]
if not hasattr(_anytran_sr, "stream_rtsp_audio"):
    _anytran_sr.stream_rtsp_audio = MagicMock()

_anytran_cl = sys.modules["anytran.chatlog"]
for _fn in ("ChatLogger", "extract_ip_from_rtsp_url"):
    if not hasattr(_anytran_cl, _fn):
        setattr(_anytran_cl, _fn, MagicMock())

_anytran_vm = sys.modules["anytran.voice_matcher"]
for _fn in (
    "extract_voice_features", "select_best_piper_voice",
):
    if not hasattr(_anytran_vm, _fn):
        setattr(_anytran_vm, _fn, MagicMock())

# Remove any empty stubs that earlier test files may have placed for modules we
# need to import as real implementations (processing.py, text_translator.py,
# utils.py, and run_file_input.py).  Keeping the other runner stubs avoids
# re-executing their heavy source files.
for _key in list(sys.modules.keys()):
    if _key in (
        "anytran.processing",
        "anytran.text_translator",
        "anytran.utils",
        "anytran.runners",
        "anytran.runners.run_file_input",
    ):
        del sys.modules[_key]

# ============================================================================
# END STUB SETUP
# ============================================================================

import numpy as np
from anytran.processing import process_audio_chunk, build_output_prefix
from anytran.runners.run_file_input import run_file_input
from anytran import text_translator


# ---------------------------------------------------------------------------
# Tests for process_audio_chunk
# ---------------------------------------------------------------------------

class TestPipelineStages(unittest.TestCase):
    def _audio_chunk(self, length=16000):
        return np.ones(length, dtype=np.float32) * 0.1

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    @patch("anytran.processing.synthesize_tts_pcm")
    @patch("anytran.processing.play_output")
    def test_stage1_only(self, mock_play, mock_tts, mock_translate_text, mock_translate_audio):
        """Stage 1 only: transcription without translation produces English output."""
        mock_translate_audio.return_value = (self._audio_chunk(), "hello world", "en")

        output = process_audio_chunk(
            self._audio_chunk(),
            16000,
            input_lang="en",
            output_lang="en",
            magnitude_threshold=0.001,
            model="tiny",
            verbose=False,
            mqtt_broker=None,
            mqtt_port=None,
            mqtt_username=None,
            mqtt_password=None,
            mqtt_topic=None,
            scribe_vad=False,
            timers=False,
            text_translation_target=None,
            slate_backend="googletrans",
            voice_lang=None,
            lang_prefix=True,
        )

        self.assertTrue(output["output"].startswith("English: "))
        self.assertIn("hello world", output["output"])
        mock_translate_text.assert_not_called()
        mock_tts.assert_not_called()
        mock_play.assert_not_called()

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    def test_stage2_translation(self, mock_translate_text, mock_translate_audio):
        """Stage 2: translated text produces target-language prefix."""
        mock_translate_audio.return_value = (self._audio_chunk(), "hello", "en")
        mock_translate_text.return_value = "bonjour"

        output = process_audio_chunk(
            self._audio_chunk(),
            16000,
            input_lang="en",
            output_lang="fr",
            magnitude_threshold=0.001,
            model="tiny",
            verbose=False,
            mqtt_broker=None,
            mqtt_port=None,
            mqtt_username=None,
            mqtt_password=None,
            mqtt_topic=None,
            scribe_vad=False,
            timers=False,
            text_translation_target="fr",
            slate_backend="googletrans",
            voice_lang=None,
            lang_prefix=True,
        )

        self.assertTrue(output["output"].startswith("French: "))
        self.assertIn("bonjour", output["output"])
        mock_translate_text.assert_called_once()

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    @patch("anytran.processing.synthesize_tts_pcm_with_cloning")
    def test_stage3_tts(self, mock_tts, mock_translate_text, mock_translate_audio):
        """Stage 3: TTS is synthesized and appended to the slate segments list."""
        mock_translate_audio.return_value = (self._audio_chunk(), "hello", "en")
        mock_translate_text.return_value = "bonjour"
        mock_tts.return_value = np.zeros(16000, dtype=np.int16)

        segments = []
        output = process_audio_chunk(
            self._audio_chunk(),
            16000,
            input_lang="en",
            output_lang="fr",
            magnitude_threshold=0.001,
            model="tiny",
            verbose=False,
            mqtt_broker=None,
            mqtt_port=None,
            mqtt_username=None,
            mqtt_password=None,
            mqtt_topic=None,
            scribe_vad=False,
            timers=False,
            text_translation_target="fr",
            slate_backend="googletrans",
            voice_lang=None,
            lang_prefix=True,
            slate_tts_segments=segments,
        )

        self.assertTrue(output["output"].startswith("French: "))
        self.assertEqual(len(segments), 1)
        mock_tts.assert_called_once()

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    @patch("anytran.processing.synthesize_tts_pcm_with_cloning")
    def test_stage3_tts_runs_when_target_is_english(self, mock_tts, mock_translate_text, mock_translate_audio):
        """TTS should run even when translation target is English."""
        mock_translate_audio.return_value = (self._audio_chunk(), "hello", "en")
        mock_translate_text.return_value = None  # Translation returns None when target is English (Stage 2 skipped)
        mock_tts.return_value = np.zeros(16000, dtype=np.int16)

        segments = []
        output = process_audio_chunk(
            self._audio_chunk(),
            16000,
            input_lang="en",
            output_lang="en",
            magnitude_threshold=0.001,
            model="tiny",
            verbose=False,
            mqtt_broker=None,
            mqtt_port=None,
            mqtt_username=None,
            mqtt_password=None,
            mqtt_topic=None,
            scribe_vad=False,
            timers=False,
            text_translation_target="en",
            slate_backend="googletrans",
            voice_lang=None,
            lang_prefix=False,
            slate_tts_segments=segments,
        )

        self.assertEqual(len(segments), 1)
        mock_tts.assert_called_once()
        self.assertEqual(output["final_lang"], "en")


class TestProcessAudioChunkOutput(unittest.TestCase):
    """Tests for the dict return value and silence handling of process_audio_chunk."""

    def _audio_chunk(self, length=16000):
        return np.ones(length, dtype=np.float32) * 0.1

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    def test_returns_dict_with_expected_keys(self, mock_translate_text, mock_translate_audio):
        """process_audio_chunk returns a dict with output/scribe/slate/final_lang keys."""
        mock_translate_audio.return_value = (self._audio_chunk(), "hello world", "en")

        result = process_audio_chunk(
            self._audio_chunk(), 16000,
            input_lang="en", output_lang="en",
            magnitude_threshold=0.001, model="tiny", verbose=False,
            mqtt_broker=None, mqtt_port=None, mqtt_username=None,
            mqtt_password=None, mqtt_topic=None,
            scribe_vad=False, timers=False,
            text_translation_target=None, slate_backend="googletrans",
            voice_lang=None,
        )

        self.assertIsInstance(result, dict)
        for key in ("output", "scribe", "slate", "final_lang"):
            self.assertIn(key, result)

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    def test_returns_none_for_silence(self, mock_translate_text, mock_translate_audio):
        """process_audio_chunk returns None when audio is below the magnitude threshold."""
        silent_audio = np.zeros(16000, dtype=np.float32)

        result = process_audio_chunk(
            silent_audio, 16000,
            input_lang="en", output_lang="en",
            magnitude_threshold=0.5,
            model="tiny", verbose=False,
            mqtt_broker=None, mqtt_port=None, mqtt_username=None,
            mqtt_password=None, mqtt_topic=None,
            scribe_vad=False, timers=False,
            text_translation_target=None, slate_backend="googletrans",
            voice_lang=None,
        )

        self.assertIsNone(result)
        mock_translate_audio.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for run_file_input (text file path)
# ---------------------------------------------------------------------------

class TestFileInputText(unittest.TestCase):
    @patch("anytran.runners.run_file_input.translate_text")
    @patch("anytran.runners.run_file_input.synthesize_tts_pcm")
    @patch("anytran.runners.run_file_input.play_output")
    @patch("anytran.runners.run_file_input.send_mqtt_text")
    @patch("anytran.runners.run_file_input.output_audio")
    def test_text_file_translation(self, mock_output_audio, mock_send, mock_play, mock_tts, mock_translate_text):
        """Translating a French .txt file writes Spanish output to slate_text_file."""
        mock_translate_text.side_effect = ["hello", "hola"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            output_path = os.path.join(tmpdir, "out.txt")
            with open(input_path, mode="w", encoding="utf-8") as fh:
                fh.write("bonjour")

            run_file_input(
                input_path,
                input_lang="fr",
                output_lang="es",
                slate_text_file=output_path,
                magnitude_threshold=0.001,
                output_audio_path=None,
                model="tiny",
                verbose=False,
                mqtt_broker=None,
                mqtt_port=1883,
                mqtt_username=None,
                mqtt_password=None,
                mqtt_topic="translation",
                scribe_vad=False,
                voice_backend="gtts",
                voice_model=None,
                window_seconds=5.0,
                overlap_seconds=0.0,
                timers=False,
                scribe_backend="auto",
                text_translation_target="es",
                slate_backend="googletrans",
                voice_lang=None,
                lang_prefix=True,
            )

            with open(output_path, mode="r", encoding="utf-8") as fh:
                content = fh.read()

        self.assertIn("Spanish: hola", content)
        mock_output_audio.assert_not_called()
        mock_play.assert_not_called()
        mock_tts.assert_not_called()
        mock_send.assert_not_called()

    @patch("anytran.runners.run_file_input.translate_text")
    @patch("anytran.runners.run_file_input.synthesize_tts_pcm")
    @patch("anytran.runners.run_file_input.play_output")
    @patch("anytran.runners.run_file_input.send_mqtt_text")
    @patch("anytran.runners.run_file_input.output_audio")
    def test_text_file_translation_no_prefix(self, mock_output_audio, mock_send, mock_play, mock_tts, mock_translate_text):
        """Without lang_prefix the slate_text_file contains plain translated text."""
        mock_translate_text.side_effect = ["hello", "hola"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            output_path = os.path.join(tmpdir, "out.txt")
            with open(input_path, mode="w", encoding="utf-8") as fh:
                fh.write("bonjour")

            run_file_input(
                input_path,
                input_lang="fr",
                output_lang="es",
                slate_text_file=output_path,
                magnitude_threshold=0.001,
                output_audio_path=None,
                model="tiny",
                verbose=False,
                mqtt_broker=None,
                mqtt_port=1883,
                mqtt_username=None,
                mqtt_password=None,
                mqtt_topic="translation",
                scribe_vad=False,
                voice_backend="gtts",
                voice_model=None,
                window_seconds=5.0,
                overlap_seconds=0.0,
                timers=False,
                scribe_backend="auto",
                text_translation_target="es",
                slate_backend="googletrans",
                voice_lang=None,
            )

            with open(output_path, mode="r", encoding="utf-8") as fh:
                content = fh.read()

        # normalize_text capitalises the first letter: "hola" → "Hola"
        self.assertIn("Hola", content)
        self.assertNotIn("Spanish: ", content)
        self.assertNotIn("English: ", content)
        self.assertNotIn("French: ", content)


# ---------------------------------------------------------------------------
# Tests for the lang_prefix option via process_audio_chunk
# ---------------------------------------------------------------------------

class TestLangPrefixOption(unittest.TestCase):
    def _audio_chunk(self, length=16000):
        return np.ones(length, dtype=np.float32) * 0.1

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    def test_no_prefix_by_default(self, mock_translate_text, mock_translate_audio):
        """Without lang_prefix the output is the raw transcription text."""
        mock_translate_audio.return_value = (self._audio_chunk(), "hello world", "en")

        output = process_audio_chunk(
            self._audio_chunk(),
            16000,
            input_lang="en",
            output_lang="en",
            magnitude_threshold=0.001,
            model="tiny",
            verbose=False,
            mqtt_broker=None,
            mqtt_port=None,
            mqtt_username=None,
            mqtt_password=None,
            mqtt_topic=None,
            scribe_vad=False,
            timers=False,
            text_translation_target=None,
            slate_backend="googletrans",
            voice_lang=None,
        )

        self.assertEqual(output["output"], "hello world")
        self.assertFalse(output["output"].startswith("English: "))

    @patch("anytran.processing.translate_audio")
    @patch("anytran.processing.translate_text")
    def test_prefix_when_enabled(self, mock_translate_text, mock_translate_audio):
        """With lang_prefix=True the output is prefixed with the language name."""
        mock_translate_audio.return_value = (self._audio_chunk(), "hello world", "en")

        output = process_audio_chunk(
            self._audio_chunk(),
            16000,
            input_lang="en",
            output_lang="en",
            magnitude_threshold=0.001,
            model="tiny",
            verbose=False,
            mqtt_broker=None,
            mqtt_port=None,
            mqtt_username=None,
            mqtt_password=None,
            mqtt_topic=None,
            scribe_vad=False,
            timers=False,
            text_translation_target=None,
            slate_backend="googletrans",
            voice_lang=None,
            lang_prefix=True,
        )

        self.assertTrue(output["output"].startswith("English: "))
        self.assertIn("hello world", output["output"])


# ---------------------------------------------------------------------------
# Tests for the scribe/slate dual-output feature in run_file_input
# ---------------------------------------------------------------------------

class TestScribeSlateOutput(unittest.TestCase):
    """run_file_input writes Stage-1 (English) text to scribe_text_file
    and Stage-2 (translated) text to slate_text_file."""

    @patch("anytran.runners.run_file_input.translate_text")
    def test_scribe_file_gets_english_text(self, mock_translate_text):
        """scribe_text_file receives the English (Stage-1) text."""
        mock_translate_text.side_effect = ["hello", "hola"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            scribe_path = os.path.join(tmpdir, "scribe.txt")
            with open(input_path, mode="w", encoding="utf-8") as fh:
                fh.write("bonjour")

            run_file_input(
                input_path,
                input_lang="fr",
                output_lang="es",
                scribe_text_file=scribe_path,
                magnitude_threshold=0.001,
                output_audio_path=None,
                mqtt_broker=None,
                text_translation_target="es",
                slate_backend="googletrans",
                lang_prefix=False,
            )

            with open(scribe_path, mode="r", encoding="utf-8") as fh:
                content = fh.read()

        # normalize_text capitalises first letter: "hello" → "Hello"
        self.assertIn("Hello", content)

    @patch("anytran.runners.run_file_input.translate_text")
    def test_slate_file_gets_translated_text(self, mock_translate_text):
        """slate_text_file receives the translated (Stage-2) text."""
        mock_translate_text.side_effect = ["hello", "hola"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            slate_path = os.path.join(tmpdir, "slate.txt")
            with open(input_path, mode="w", encoding="utf-8") as fh:
                fh.write("bonjour")

            run_file_input(
                input_path,
                input_lang="fr",
                output_lang="es",
                slate_text_file=slate_path,
                magnitude_threshold=0.001,
                output_audio_path=None,
                mqtt_broker=None,
                text_translation_target="es",
                slate_backend="googletrans",
                lang_prefix=False,
            )

            with open(slate_path, mode="r", encoding="utf-8") as fh:
                content = fh.read()

        # normalize_text capitalises first letter: "hola" → "Hola"
        self.assertIn("Hola", content)


# ---------------------------------------------------------------------------
# Tests for the MarianMT translation backend
# ---------------------------------------------------------------------------

class TestMarianMTBackend(unittest.TestCase):
    """Unit tests for the MarianMT translation backend."""

    def setUp(self):
        # Reset cached model state before each test
        text_translator._marianmt_model = None
        text_translator._marianmt_tokenizer = None
        text_translator._marianmt_loaded_model_name = None
        text_translator._marianmt_model_name = None  # default: auto-derive from language pair

    @patch("anytran.text_translator.MarianTokenizer")
    @patch("anytran.text_translator.MarianMTModel")
    def test_translate_text_marianmt_returns_translation(self, mock_model_cls, mock_tokenizer_cls):
        """translate_text_marianmt should return the decoded translation."""
        mock_inputs = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "bonjour"

        result = text_translator.translate_text_marianmt("hello", "en", "fr", verbose=False)

        self.assertEqual(result, "bonjour")
        # Model should have been loaded with the auto-derived name
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("Helsinki-NLP/opus-mt-en-fr")
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)

    @patch("anytran.text_translator.MarianTokenizer")
    @patch("anytran.text_translator.MarianMTModel")
    def test_auto_derives_model_for_reverse_direction(self, mock_model_cls, mock_tokenizer_cls):
        """When source/target are reversed the correct reverse model is auto-derived."""
        mock_inputs = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "hello"

        result = text_translator.translate_text_marianmt(
            "Bonjour", "fr", "en", verbose=False
        )

        self.assertEqual(result, "hello")
        # Should use fr-en model, NOT en-ROMANCE
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("Helsinki-NLP/opus-mt-fr-en")

    @patch("anytran.text_translator.MarianTokenizer")
    @patch("anytran.text_translator.MarianMTModel")
    def test_explicit_model_overrides_auto_derive(self, mock_model_cls, mock_tokenizer_cls):
        """set_marianmt_config should override auto-derivation."""
        text_translator.set_marianmt_config("Helsinki-NLP/opus-mt-en-ROMANCE")
        mock_inputs = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "bonjour"

        text_translator.translate_text_marianmt("hello", "en", "fr", verbose=False)

        mock_tokenizer_cls.from_pretrained.assert_called_once_with("Helsinki-NLP/opus-mt-en-ROMANCE")

    @patch("anytran.text_translator.MarianTokenizer")
    @patch("anytran.text_translator.MarianMTModel")
    def test_translate_text_marianmt_verbose(self, mock_model_cls, mock_tokenizer_cls):
        """translate_text_marianmt should not raise when verbose=True."""
        mock_inputs = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "hola"

        result = text_translator.translate_text_marianmt("hello", "en", "es", verbose=True)

        self.assertEqual(result, "hola")

    @patch("anytran.text_translator.MarianMTModel")
    @patch("anytran.text_translator.MarianTokenizer")
    def test_translate_text_marianmt_exception_returns_none(self, mock_tokenizer_cls, mock_model_cls):
        """translate_text_marianmt should return None when an exception occurs."""
        mock_tokenizer_cls.from_pretrained.side_effect = Exception("model not found")

        result = text_translator.translate_text_marianmt("hello", "en", "fr", verbose=False)

        self.assertIsNone(result)

    @patch("anytran.text_translator.MarianTokenizer")
    @patch("anytran.text_translator.MarianMTModel")
    def test_set_marianmt_config(self, mock_model_cls, mock_tokenizer_cls):
        """set_marianmt_config should update the model name."""
        text_translator.set_marianmt_config("Helsinki-NLP/opus-mt-en-de")
        self.assertEqual(text_translator._marianmt_model_name, "Helsinki-NLP/opus-mt-en-de")

    def test_translate_text_dispatcher_marianmt(self):
        """translate_text() with backend='marianmt' should call translate_text_marianmt."""
        with patch("anytran.text_translator.translate_text_marianmt", return_value="hallo") as mock_fn:
            result = text_translator.translate_text("hello", "en", "de", backend="marianmt")
            mock_fn.assert_called_once_with("hello", "en", "de", False)
            self.assertEqual(result, "hallo")


# ---------------------------------------------------------------------------
# Tests for zh -> zh-cn mapping in googletrans backend
# ---------------------------------------------------------------------------

class TestGoogletransZhMapping(unittest.TestCase):
    """translate_text_googletrans must map bare 'zh' target_lang to 'zh-cn'."""

    @patch("anytran.text_translator._get_googletrans_translator")
    def test_zh_mapped_to_zh_cn(self, mock_get_translator):
        """When target_lang is 'zh', googletrans should be called with 'zh-cn'."""
        mock_translator = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "你好"
        mock_translator.translate.return_value = mock_result
        mock_get_translator.return_value = mock_translator

        text_translator._GOOGLETRANS_AVAILABLE = True
        result = text_translator.translate_text_googletrans("hello", "en", "zh")

        self.assertEqual(result, "你好")
        call_kwargs = mock_translator.translate.call_args
        self.assertEqual(call_kwargs.kwargs.get("dest") or call_kwargs[1].get("dest"), "zh-cn")

    @patch("anytran.text_translator._get_googletrans_translator")
    def test_zh_cn_unchanged(self, mock_get_translator):
        """When target_lang is already 'zh-cn', it should pass through unchanged."""
        mock_translator = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "你好"
        mock_translator.translate.return_value = mock_result
        mock_get_translator.return_value = mock_translator

        text_translator._GOOGLETRANS_AVAILABLE = True
        result = text_translator.translate_text_googletrans("hello", "en", "zh-cn")

        self.assertEqual(result, "你好")
        call_kwargs = mock_translator.translate.call_args
        self.assertEqual(call_kwargs.kwargs.get("dest") or call_kwargs[1].get("dest"), "zh-cn")

    @patch("anytran.text_translator._get_googletrans_translator")
    def test_zh_tw_unchanged(self, mock_get_translator):
        """When target_lang is 'zh-tw', it should pass through unchanged."""
        mock_translator = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "你好"
        mock_translator.translate.return_value = mock_result
        mock_get_translator.return_value = mock_translator

        text_translator._GOOGLETRANS_AVAILABLE = True
        result = text_translator.translate_text_googletrans("hello", "en", "zh-tw")

        self.assertEqual(result, "你好")
        call_kwargs = mock_translator.translate.call_args
        self.assertEqual(call_kwargs.kwargs.get("dest") or call_kwargs[1].get("dest"), "zh-tw")


if __name__ == "__main__":
    unittest.main()
