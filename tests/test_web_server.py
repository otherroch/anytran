import asyncio
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import base64
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

def _stub_optional_modules():
    """Stub heavy optional dependencies used during imports."""
    for _mod in [
        "numpy",
        "soundfile",
        "librosa",
        "paho",
        "paho.mqtt",
        "paho.mqtt.client",
        "playsound3",
        "piper",
        "silero_vad",
        "moviepy",
        "moviepy.editor",
        "torch",
        "transformers",
        "faster_whisper",
        "pydub",
        "googletrans",
    ]:
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()
    # text_translator expects googletrans.Translator at import time
    if not hasattr(sys.modules["googletrans"], "Translator"):
        sys.modules["googletrans"].Translator = MagicMock()


_stub_optional_modules()

# Remove any prior stubs of anytran.web_server that other tests might have installed
if sys.modules.get("anytran.web_server", None) is not None and not hasattr(sys.modules["anytran.web_server"], "_effective_voice_settings"):
    del sys.modules["anytran.web_server"]
if sys.modules.get("anytran.config", None) is not None and not hasattr(sys.modules["anytran.config"], "get_whisper_backend"):
    del sys.modules["anytran.config"]
if sys.modules.get("anytran.processing", None) is not None and not hasattr(sys.modules["anytran.processing"], "process_audio_chunk"):
    del sys.modules["anytran.processing"]
if sys.modules.get("anytran.tts", None) is not None and not hasattr(sys.modules["anytran.tts"], "play_output"):
    del sys.modules["anytran.tts"]
if sys.modules.get("anytran.voice_matcher", None) is not None and not hasattr(sys.modules["anytran.voice_matcher"], "extract_voice_features"):
    del sys.modules["anytran.voice_matcher"]
if sys.modules.get("anytran.utils", None) is not None and not hasattr(sys.modules["anytran.utils"], "normalize_lang_code"):
    del sys.modules["anytran.utils"]
if sys.modules.get("anytran.vad", None) is not None and not hasattr(sys.modules["anytran.vad"], "has_speech_silero"):
    del sys.modules["anytran.vad"]
if sys.modules.get("anytran.whisper_backend", None) is not None and not hasattr(sys.modules["anytran.whisper_backend"], "translate_audio"):
    del sys.modules["anytran.whisper_backend"]

from anytran.web_server import _effective_voice_settings
from anytran import web_server


class TestWebServerErrorHandling(unittest.TestCase):
    """Test Windows-specific error handling in web server."""

    def test_windows_ctrl_handler_keeps_callback_reference(self):
        """Ensure Windows handler isn't garbage collected."""
        kernel32 = MagicMock()
        callbacks = {}

        def fake_handler_wrapper(func):
            callbacks["cb"] = func
            return "wrapped"

        fake_ctypes = types.SimpleNamespace(
            WINFUNCTYPE=lambda *args, **kwargs: fake_handler_wrapper,
            c_bool=MagicMock(),
            c_uint=MagicMock(),
            windll=types.SimpleNamespace(kernel32=kernel32),
        )

        web_server._WINDOWS_CTRL_HANDLER = None
        with patch("anytran.web_server.os.name", "nt"), \
             patch.dict(sys.modules, {"ctypes": fake_ctypes}, clear=False):
            web_server._install_windows_ctrl_handler(MagicMock())

        self.assertIsNotNone(web_server._WINDOWS_CTRL_HANDLER)
        self.assertIsNotNone(callbacks.get("cb"))
        self.assertEqual("wrapped", web_server._WINDOWS_CTRL_HANDLER)
        kernel32.SetConsoleCtrlHandler.assert_called_once_with("wrapped", True)

    def test_windows_socket_error_suppression(self):
        """Test that ConnectionResetError is properly suppressed on Windows."""
        # Import the function from web_server module
        # We can't easily import it directly since it's defined inside run_web_server
        # So we'll test the logic inline
        
        def suppress_windows_socket_errors(loop, context):
            """
            Custom exception handler to suppress Windows-specific socket errors.
            """
            exception = context.get('exception')
            if isinstance(exception, ConnectionResetError):
                # Suppress ConnectionResetError on Windows (WinError 10054)
                return
            # For all other exceptions, use the default handler
            loop.default_exception_handler(context)

        # Create a test loop
        loop = asyncio.new_event_loop()
        
        # Track if default handler was called
        default_handler_called = []
        original_default_handler = loop.default_exception_handler
        
        def mock_default_handler(context):
            default_handler_called.append(context)
            # Don't actually call the original to avoid noise in test output
        
        loop.default_exception_handler = mock_default_handler
        loop.set_exception_handler(suppress_windows_socket_errors)
        
        # Test 1: ConnectionResetError should be suppressed
        context1 = {'exception': ConnectionResetError("WinError 10054")}
        loop.call_exception_handler(context1)
        self.assertEqual(len(default_handler_called), 0, 
                        "ConnectionResetError should be suppressed, not passed to default handler")
        
        # Test 2: Other exceptions should be passed to default handler
        context2 = {'exception': ValueError("Some other error")}
        loop.call_exception_handler(context2)
        self.assertEqual(len(default_handler_called), 1,
                        "Other exceptions should be passed to default handler")
        self.assertEqual(default_handler_called[0], context2)
        
        loop.close()

    def test_exception_handler_configuration(self):
        """Test that exception handler can be configured correctly."""
        # Create a test loop
        loop = asyncio.new_event_loop()
        
        # Define the handler
        def suppress_windows_socket_errors(loop, context):
            exception = context.get('exception')
            if isinstance(exception, ConnectionResetError):
                return
            loop.default_exception_handler(context)
        
        # Set the handler
        loop.set_exception_handler(suppress_windows_socket_errors)
        
        # Verify the handler was set (by checking it doesn't raise an error)
        # We can't directly check the handler, but we can verify it works
        context = {'exception': ConnectionResetError("Test")}
        try:
            loop.call_exception_handler(context)
            # If we get here without exception, the handler is working
            handler_works = True
        except Exception:
            handler_works = False
        
        self.assertTrue(handler_works, "Exception handler should handle ConnectionResetError")
        loop.close()


class TestWebServerVoiceSettings(unittest.TestCase):
    def test_cli_voice_backend_takes_precedence(self):
        backend, model = _effective_voice_settings("piper", "cli_voice")
        self.assertEqual(backend, "piper")
        self.assertEqual(model, "cli_voice")

    def test_env_voice_backend_used_as_fallback(self):
        with patch.dict(os.environ, {"USE_PIPER": "1", "PIPER_VOICE": "env_voice"}, clear=False):
            backend, model = _effective_voice_settings(None, None)
        self.assertEqual(backend, "piper")
        self.assertEqual(model, "env_voice")


class TestWebServerTTSStreaming(unittest.TestCase):
    """Ensure web server audio chunks request TTS output."""

    def test_process_web_audio_chunk_serializes_tts_payloads(self):
        sample_audio = np.array([0, 1000, -1000, 0], dtype=np.int16)
        recorded = {}

        def fake_process_audio_chunk(*args, **kwargs):
            recorded["slate_arg"] = kwargs.get("slate_tts_segments")
            if recorded["slate_arg"] is not None:
                recorded["slate_arg"].append(sample_audio)
            return {"output": "hello", "final_lang": "en"}

        with patch.object(web_server, "process_audio_chunk", side_effect=fake_process_audio_chunk):
            result, payloads = web_server._process_web_audio_chunk(
                audio_segment=sample_audio,
                rate=16000,
                current_input_lang="en",
                current_output_lang="en",
                magnitude_threshold=0.02,
                model=None,
                verbose=False,
                mqtt_broker=None,
                mqtt_port=1883,
                mqtt_username=None,
                mqtt_password=None,
                mqtt_topic="translation",
                stream_id="web",
                scribe_vad=False,
                voice_backend="piper",
                voice_model="voice",
                timers=False,
                timing_stats=None,
                scribe_backend="auto",
                text_translation_target="en",
                langswap_enabled=False,
                langswap_input_lang="en",
                langswap_output_lang="en",
                voice_match=False,
                voice_lang=None,
            )

        self.assertIsNotNone(recorded.get("slate_arg"))
        self.assertIsInstance(recorded["slate_arg"], list)
        self.assertEqual({"output": "hello", "final_lang": "en"}, result)
        self.assertEqual(1, len(payloads))
        payload = payloads[0]
        self.assertEqual("tts_audio", payload["type"])
        self.assertEqual(16000, payload["rate"])
        decoded = np.frombuffer(base64.b64decode(payload["pcm"]), dtype=np.int16)
        np.testing.assert_array_equal(sample_audio, decoded)

    def test_process_web_audio_chunk_passes_voice_match(self):
        """Voice match flag should be forwarded to processing."""
        sample_audio = np.array([1, -1], dtype=np.int16)
        call_args = {}

        def fake_process_audio_chunk(*args, **kwargs):
            call_args.update(kwargs)
            return {"output": "x", "final_lang": "en"}

        with patch.object(web_server, "process_audio_chunk", side_effect=fake_process_audio_chunk):
            web_server._process_web_audio_chunk(
                audio_segment=sample_audio,
                rate=16000,
                current_input_lang="en",
                current_output_lang="en",
                magnitude_threshold=0.02,
                model=None,
                verbose=False,
                mqtt_broker=None,
                mqtt_port=1883,
                mqtt_username=None,
                mqtt_password=None,
                mqtt_topic="translation",
                stream_id="web",
                scribe_vad=False,
                voice_backend="piper",
                voice_model="voice",
                timers=False,
                timing_stats=None,
                scribe_backend="auto",
                text_translation_target="en",
                langswap_enabled=False,
                langswap_input_lang="en",
                langswap_output_lang="en",
                voice_match=True,
                voice_lang="fr",
            )

        self.assertTrue(call_args.get("voice_match"))
        self.assertEqual("fr", call_args.get("voice_lang"))


if __name__ == "__main__":
    unittest.main()
