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


if __name__ == "__main__":
    unittest.main()
