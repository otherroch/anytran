import os
from dataclasses import dataclass

# Existing config code...


@dataclass
class CommonConfig:
    """Common configuration parameters shared across all runners.

    Parameters are optional; ``None`` indicates that the default value or the
    caller's explicit argument should be used.  This class is intended to
    reduce the number of parameters that need to travel through the call
    graph.  Values that rarely change (e.g. language, MQTT broker, VAD
    settings, TTS backend) can be set once here.
    """
    input_lang: str | None = None
    output_lang: str | None = None
    magnitude_threshold: float | None = None
    model: str | None = None
    verbose: bool | None = None
    mqtt_broker: str | None = None
    mqtt_port: int | None = None
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_topic: str | None = None
    scribe_vad: bool | None = None
    voice_backend: str | None = None
    voice_model: str | None = None
    timers: bool | None = None
    scribe_backend: str | None = None
    text_translation_target: str | None = None
    slate_backend: str | None = None
    voice_lang: str | None = None
    lang_prefix: bool | None = None
    # Additional optional params can be added as needed.

# Existing whisper and gemma4 config code...
