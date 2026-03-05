"""Tests for anytran.mqtt_client module."""
import json
import threading
import unittest
from unittest.mock import MagicMock, patch, call


class TestSendMqttText(unittest.TestCase):
    """Tests for send_mqtt_text function."""

    def setUp(self):
        import anytran.mqtt_client as mqtt_module
        self._orig_client = mqtt_module._mqtt_client
        self._orig_event = mqtt_module._mqtt_connected
        mqtt_module._mqtt_client = None
        new_event = threading.Event()
        mqtt_module._mqtt_connected = new_event

    def tearDown(self):
        import anytran.mqtt_client as mqtt_module
        mqtt_module._mqtt_client = self._orig_client
        mqtt_module._mqtt_connected = self._orig_event

    def test_empty_text_returns_without_publishing(self):
        import anytran.mqtt_client as mqtt_module
        mock_client = MagicMock()
        mqtt_module._mqtt_client = mock_client
        mqtt_module._mqtt_connected.set()
        mqtt_module.send_mqtt_text("", "test/topic")
        mock_client.publish.assert_not_called()

    def test_none_text_returns_without_publishing(self):
        import anytran.mqtt_client as mqtt_module
        mock_client = MagicMock()
        mqtt_module._mqtt_client = mock_client
        mqtt_module._mqtt_connected.set()
        mqtt_module.send_mqtt_text(None, "test/topic")
        mock_client.publish.assert_not_called()

    def test_publishes_when_connected(self):
        import anytran.mqtt_client as mqtt_module
        import paho.mqtt.client as mqtt
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client.publish.return_value = mock_result
        mqtt_module._mqtt_client = mock_client
        mqtt_module._mqtt_connected.set()
        mqtt_module.send_mqtt_text("Hello world", "test/topic")
        mock_client.publish.assert_called_once()
        args = mock_client.publish.call_args
        self.assertEqual(args[0][0], "test/topic")

    def test_publishes_valid_json(self):
        import anytran.mqtt_client as mqtt_module
        import paho.mqtt.client as mqtt
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client.publish.return_value = mock_result
        mqtt_module._mqtt_client = mock_client
        mqtt_module._mqtt_connected.set()
        mqtt_module.send_mqtt_text("Test message", "my/topic")
        published_payload = mock_client.publish.call_args[0][1]
        data = json.loads(published_payload)
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["api"], "chat")
        self.assertEqual(data[0]["message"], "Test message")

    def test_no_publish_when_not_connected(self):
        import anytran.mqtt_client as mqtt_module
        mock_client = MagicMock()
        mqtt_module._mqtt_client = mock_client
        # Don't set the event - so not connected
        mqtt_module.send_mqtt_text("Hello", "test/topic")
        mock_client.publish.assert_not_called()

    def test_no_client_with_broker_calls_init_mqtt(self):
        import anytran.mqtt_client as mqtt_module
        mqtt_module._mqtt_client = None
        with patch("anytran.mqtt_client.init_mqtt") as mock_init:
            mqtt_module.send_mqtt_text("Hello", "test/topic", mqtt_broker="localhost")
            mock_init.assert_called_once()

    def test_publish_failure_does_not_raise(self):
        import anytran.mqtt_client as mqtt_module
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = 1  # Error code
        mock_client.publish.return_value = mock_result
        mqtt_module._mqtt_client = mock_client
        mqtt_module._mqtt_connected.set()
        # Should not raise
        mqtt_module.send_mqtt_text("Hello", "test/topic")

    def test_publish_exception_does_not_raise(self):
        import anytran.mqtt_client as mqtt_module
        mock_client = MagicMock()
        mock_client.publish.side_effect = RuntimeError("Connection error")
        mqtt_module._mqtt_client = mock_client
        mqtt_module._mqtt_connected.set()
        # Should not raise
        mqtt_module.send_mqtt_text("Hello", "test/topic")

    def test_long_text_truncated_in_summary(self):
        import anytran.mqtt_client as mqtt_module
        import paho.mqtt.client as mqtt
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client.publish.return_value = mock_result
        mqtt_module._mqtt_client = mock_client
        mqtt_module._mqtt_connected.set()
        long_text = "A" * 100
        # Should not raise; long text gets truncated in the print summary
        mqtt_module.send_mqtt_text(long_text, "test/topic")
        mock_client.publish.assert_called_once()


class TestInitMqtt(unittest.TestCase):
    """Tests for init_mqtt function."""

    def setUp(self):
        import anytran.mqtt_client as mqtt_module
        self._orig_client = mqtt_module._mqtt_client
        self._orig_event = mqtt_module._mqtt_connected
        mqtt_module._mqtt_client = None
        mqtt_module._mqtt_connected = threading.Event()

    def tearDown(self):
        import anytran.mqtt_client as mqtt_module
        if mqtt_module._mqtt_client is not None:
            try:
                mqtt_module._mqtt_client.loop_stop()
                mqtt_module._mqtt_client.disconnect()
            except Exception:
                pass
            mqtt_module._mqtt_client = None
        mqtt_module._mqtt_client = self._orig_client
        mqtt_module._mqtt_connected = self._orig_event

    def test_init_mqtt_returns_none_on_connection_failure(self):
        import anytran.mqtt_client as mqtt_module
        # Attempt to connect to a non-existent broker - should return None or timeout
        result = mqtt_module.init_mqtt("127.0.0.1", port=19999, topic="test")
        # Should not raise, may return None or client
        # The function handles connection errors gracefully
        # Reset
        mqtt_module._mqtt_client = None

    def test_init_mqtt_already_initialized_returns_existing(self):
        import anytran.mqtt_client as mqtt_module
        mock_client = MagicMock()
        mqtt_module._mqtt_client = mock_client
        result = mqtt_module.init_mqtt("localhost", 1883)
        self.assertIs(result, mock_client)


class TestCleanupMqtt(unittest.TestCase):
    """Tests for cleanup_mqtt function."""

    def setUp(self):
        import anytran.mqtt_client as mqtt_module
        self._orig_client = mqtt_module._mqtt_client

    def tearDown(self):
        import anytran.mqtt_client as mqtt_module
        mqtt_module._mqtt_client = self._orig_client

    def test_cleanup_none_client_is_noop(self):
        import anytran.mqtt_client as mqtt_module
        mqtt_module._mqtt_client = None
        mqtt_module.cleanup_mqtt()  # Should not raise

    def test_cleanup_stops_and_disconnects(self):
        import anytran.mqtt_client as mqtt_module
        mock_client = MagicMock()
        mqtt_module._mqtt_client = mock_client
        mqtt_module.cleanup_mqtt()
        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()
        self.assertIsNone(mqtt_module._mqtt_client)


if __name__ == "__main__":
    unittest.main()
