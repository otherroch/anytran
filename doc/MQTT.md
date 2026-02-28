# MQTT Publishing

This document describes the options for publishing transcription/translation results to an MQTT broker.

---

## MQTT Publishing

These options enable publishing transcription/translation results to an MQTT broker.

### `--mqtt-broker <hostname|ip>`
- **Type**: Hostname or IP address
- **Example**: `redis.example.com`, `localhost`, `192.168.1.100`
- **Required**: To enable MQTT publishing
- **Purpose**: Address of MQTT broker to connect to
- **Behavior**: If set, transcription results are published to MQTT topic
- **Interactions**:
  - `--mqtt-port`: Defaults to 1883 if not specified
  - `--mqtt-username` and `--mqtt-password`: Optional authentication
  - `--mqtt-topic`: Defaults to `translation`
  - `--mqtt-topic-names`: Custom topics for each stream (if multiple RTSP)

### `--mqtt-port <int>`
- **Type**: Integer (1-65535)
- **Default**: `1883` (standard MQTT port)
- **Common ports**:
  - `1883`: Standard MQTT
  - `8883`: MQTT over TLS
- **Purpose**: Port number for MQTT broker connection
- **Interactions**:
  - Only used if `--mqtt-broker` is specified
  - Must match broker's listening port

### `--mqtt-username <string>`
- **Type**: String
- **Optional**: Yes (only needed if broker requires authentication)
- **Purpose**: Username for MQTT broker authentication
- **Behavior**: Sent with MQTT CONNECT packet
- **Interactions**:
  - Works with `--mqtt-password`
  - Both or neither should be specified for auth

### `--mqtt-password <string>`
- **Type**: String (usually API key or password)
- **Optional**: Yes (only needed if broker requires authentication)
- **Purpose**: Password/token for MQTT broker authentication
- **Security note**: Consider using environment variables instead of command line
- **Interactions**:
  - Works with `--mqtt-username`
  - Both or neither should be specified for auth

### `--mqtt-topic <string>`
- **Type**: MQTT topic path
- **Default**: `translation`
- **Example**: `audio/translation`, `whisper/en_to_fr`
- **Purpose**: MQTT topic where results are published
- **Behavior**: Each transcription/translation result published under this topic
- **Single stream**: All messages to single topic
- **Multiple streams**: Can override per-stream with `--mqtt-topic-names`
- **Interactions**:
  - Only used if `--mqtt-broker` specified
  - `--mqtt-topic-names`: Overrides for individual RTSP streams

### `--mqtt-topic-names <string>`
- **Type**: Topic name (can be repeated)
- **Usage**: `--mqtt-topic-names topic1 --mqtt-topic-names topic2`
- **Requirements**:
  - Must specify for **each** RTSP stream
  - Count must exactly match number of `--rtsp` arguments
  - Error if count mismatch
- **Purpose**: Custom MQTT topic for each RTSP stream
- **Example**:
  ```bash
  anytran --rtsp url1 --rtsp url2 \
    --mqtt-topic-names stream1_translation --mqtt-topic-names stream2_translation
  ```
- **Interactions**:
  - Only used with multiple `--rtsp` streams
  - Overrides default `--mqtt-topic`
  - Count must match RTSP count
