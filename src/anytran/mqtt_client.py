import atexit
import json
import threading

import paho.mqtt.client as mqtt

_mqtt_client = None
_mqtt_connected = threading.Event()  # Use Event for proper signaling


def init_mqtt(broker, port=1883, username=None, password=None, topic=None):
    global _mqtt_client, _mqtt_connected
    if _mqtt_client is not None:
        return _mqtt_client

    try:
        _mqtt_client = mqtt.Client()

        if username and password:
            _mqtt_client.username_pw_set(username, password)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                _mqtt_connected.set()  # Signal connection
                print(f"Connected to MQTT broker: {broker}:{port}")
            else:
                print(f"Failed to connect to MQTT broker, return code {rc}")

        def on_disconnect(client, userdata, rc):
            _mqtt_connected.clear()  # Clear connection flag
            print("Disconnected from MQTT broker")

        _mqtt_client.on_connect = on_connect
        _mqtt_client.on_disconnect = on_disconnect

        _mqtt_client.connect(broker, port, 60)
        _mqtt_client.loop_start()

        timeout = 5
        # Use Event.wait() instead of busy-waiting
        if not _mqtt_connected.wait(timeout=timeout):
            print("Warning: MQTT connection timeout")

        return _mqtt_client
    except Exception as exc:
        print(f"Error initializing MQTT: {exc}")
        return None


def send_mqtt_text(text, topic, mqtt_broker=None, mqtt_port=1883, mqtt_username=None, mqtt_password=None):
    global _mqtt_client, _mqtt_connected

    if not text:
        return

    if _mqtt_client is None and mqtt_broker:
        init_mqtt(mqtt_broker, mqtt_port, mqtt_username, mqtt_password, topic)

    if _mqtt_client and _mqtt_connected.is_set():  # Check if connected
        try:
            json_message = json.dumps([{"api": "chat", "message": text}])
            result = _mqtt_client.publish(topic, json_message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                summary = text[:50] + "..." if len(text) > 50 else text
                print(f"Published to MQTT topic '{topic}': {summary}")
            else:
                print(f"Failed to publish to MQTT, error code: {result.rc}")
        except Exception as exc:
            print(f"Error publishing to MQTT: {exc}")


def cleanup_mqtt():
    global _mqtt_client
    if _mqtt_client:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
        _mqtt_client = None


atexit.register(cleanup_mqtt)
