import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

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


class TestWebServerErrorHandling(unittest.TestCase):
    """Test Windows-specific error handling in web server."""

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


if __name__ == "__main__":
    unittest.main()
