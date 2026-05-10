from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class RunnerConfig:
    """
    Configuration for an Anytran runner session.
    Contains parameters that rarely change during a session.
    """
    input_lang: Optional[str] = None
    output_lang: Optional[str] = None
    magnitude_threshold: float = 0.02
    model: Optional[str] = None
    verbose: bool = False
    mqtt_broker: Optional[str] = None
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_topic: str = "translation"
    scribe_vad: bool = True
    voice_backend: str = "gtts"
    voice_model: Optional[str] = None
    chat_logger: Any = None
    rtsp_ip: Optional[str] = None
    timers: bool = False
    scribe_backend: str = "auto"
    text_translation_target: Optional[str] = None
    slate_backend: str = "googletrans"
    voice_lang: Optional[str] = None
    scribe_text_file: Optional[str] = None
    slate_text_file: Optional[str] = None
    langswap_enabled: bool = False
    langswap_input_lang: Optional[str] = None
    langswap_output_lang: Optional[str] = None
    voice_match: bool = False
    lang_prefix: bool = False
    normalize: bool = True
    dedup: bool = False