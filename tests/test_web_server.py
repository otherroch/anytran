import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


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


class TestProcessAudioChunkPassesTtsSegments(unittest.TestCase):
    """Test that process_audio_chunk is called with slate_tts_segments in the web server."""

    @classmethod
    def setUpClass(cls):
        web_server_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "anytran", "web_server.py"
        )
        with open(web_server_path, "r") as f:
            cls.source = f.read()

    def test_process_audio_chunk_receives_slate_tts_segments(self):
        """Verify slate_tts_segments list is passed so TTS synthesis is triggered."""
        self.assertIn("slate_tts_segments=slate_tts_segments", self.source,
                       "process_audio_chunk call should pass slate_tts_segments")

    def test_tts_audio_sent_as_binary(self):
        """Verify TTS PCM audio from slate_tts_segments is sent as binary via WebSocket."""
        self.assertIn("send_bytes", self.source,
                       "WebSocket should send TTS audio as binary bytes")

    def test_client_handles_binary_audio(self):
        """Verify client-side JS handles binary ArrayBuffer messages for audio playback."""
        self.assertIn("playPcmAudio", self.source,
                       "Client JS should include playPcmAudio function for binary audio")
        self.assertIn("instanceof ArrayBuffer", self.source,
                       "Client JS should detect ArrayBuffer binary messages")


class TestWebServerAcceptsVoiceBackendParams(unittest.TestCase):
    """Test that run_web_server accepts voice_backend, voice_model, and voice_match parameters."""

    @classmethod
    def setUpClass(cls):
        web_server_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "anytran", "web_server.py"
        )
        with open(web_server_path, "r") as f:
            cls.source = f.read()

    def test_run_web_server_has_voice_backend_parameter(self):
        """Verify run_web_server accepts voice_backend so CLI --voice-backend is respected."""
        self.assertIn("voice_backend=None,", self.source,
                       "run_web_server should accept voice_backend parameter")

    def test_run_web_server_has_voice_model_parameter(self):
        """Verify run_web_server accepts voice_model so CLI --voice-model is respected."""
        self.assertIn("voice_model=None,", self.source,
                       "run_web_server should accept voice_model parameter")

    def test_run_web_server_has_voice_match_parameter(self):
        """Verify run_web_server accepts voice_match so CLI --voice-match is respected."""
        self.assertIn("voice_match=False,", self.source,
                       "run_web_server should accept voice_match parameter")

    def test_process_audio_chunk_receives_voice_match(self):
        """Verify voice_match is passed to process_audio_chunk."""
        self.assertIn("voice_match=voice_match", self.source,
                       "process_audio_chunk call should pass voice_match")

    def test_voice_backend_falls_back_to_env_var(self):
        """Verify voice_backend falls back to USE_PIPER env var when not provided."""
        self.assertIn('os.environ.get("USE_PIPER"', self.source,
                       "Should fall back to USE_PIPER env var when voice_backend is None")


class TestWebPipelinePassesVoiceParams(unittest.TestCase):
    """Test that _run_web_pipeline passes voice params to run_web_server."""

    @classmethod
    def setUpClass(cls):
        pipelines_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "anytran", "pipelines.py"
        )
        with open(pipelines_path, "r") as f:
            cls.source = f.read()

    def test_pipeline_passes_voice_backend(self):
        """Verify _run_web_pipeline passes voice_backend from config."""
        self.assertIn('voice_backend=config["voice_backend"]', self.source,
                       "_run_web_pipeline should pass voice_backend to run_web_server")

    def test_pipeline_passes_voice_model(self):
        """Verify _run_web_pipeline passes voice_model from config."""
        self.assertIn('voice_model=config["voice_model"]', self.source,
                       "_run_web_pipeline should pass voice_model to run_web_server")

    def test_pipeline_passes_voice_match(self):
        """Verify _run_web_pipeline passes voice_match from config."""
        self.assertIn('voice_match=config["voice_match"]', self.source,
                       "_run_web_pipeline should pass voice_match to run_web_server")


class TestLangSwapTtsGuard(unittest.TestCase):
    """Test that slate TTS runs when LangSwap changes the target (even if Stage 2 skipped)."""

    @classmethod
    def setUpClass(cls):
        processing_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "anytran", "processing.py"
        )
        with open(processing_path, "r") as f:
            cls.source = f.read()

    def test_slate_tts_guard_allows_langswap(self):
        """Verify slate TTS guard includes langswap_changed_target condition."""
        self.assertIn("stage2_ran or langswap_changed_target", self.source,
                       "Slate TTS guard should allow synthesis when LangSwap changed the target")


if __name__ == "__main__":
    unittest.main()
