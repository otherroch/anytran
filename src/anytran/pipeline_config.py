"""Pipeline configuration dataclasses.

This module groups the many rarely-changing parameters that flow through the
runner -> process_audio_chunk call graph into a small number of coherent
dataclasses.  This reduces the fan-out of individual arguments and makes it
clear which settings belong together.

Groups
------
* **MQTTConfig** - MQTT broker connection and topic.
* **PipelineConfig** - All pipeline-wide settings (language, model, TTS,
  backends, timing, etc.).
* **OutputConfig** - File paths for text and audio outputs.

Usage
-----
Create config objects once per run (before the processing loop) and reuse
them for every audio chunk::

    from anytran.pipeline_config import MQTTConfig, PipelineConfig, OutputConfig

    cfg = PipelineConfig(
        input_lang="en",
        output_lang="fr",
        text_translation_target="fr",
        scribe_backend="auto",
        slate_backend="googletrans",
        voice_backend="gtts",
    )
    mqtt = MQTTConfig(broker="localhost", topic="translation")

    for chunk in audio_chunks:
        result = process_audio_chunk(chunk, rate, cfg, mqtt, stream_id="rtsp")
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# MQTT configuration
# ---------------------------------------------------------------------------

@dataclass
class MQTTConfig:
    """MQTT broker connection and topic.

    Parameters
    ----------
    broker : str or None
        MQTT broker hostname.  ``None`` disables MQTT output.
    port : int
        Broker port (default 1883).
    username : str or None
        Optional broker username.
    password : str or None
        Optional broker password.
    topic : str
        Default MQTT topic (default ``"translation"``).
    """

    broker: Optional[str] = None
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    topic: str = "translation"

    @property
    def is_enabled(self) -> bool:
        """``True`` when a broker hostname is set."""
        return self.broker is not None


# ---------------------------------------------------------------------------
# Pipeline (behaviour) configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Pipeline-wide settings that rarely change during a run.

    This consolidates the ~25+ scalar parameters that were previously passed
    individually through every layer of the call graph into a single object.
    Create one instance per runner invocation (before the processing loop) and
    pass it unchanged to every :func:`process_audio_chunk` call.

    Parameters
    ----------
    input_lang : str or None
        Source language code (e.g. ``"en"``, ``"de"``, ``"auto"``).  ``None``
        lets the backend auto-detect.
    output_lang : str or None
        Target language for transcription (Stage 1).
    text_translation_target : str or None
        Target language for Stage 2 text-to-text translation.  ``None`` or
        ``"en"`` skips Stage 2.
    model : str or None
        Speech/translation model name (e.g. ``"medium"`` for whispercpp).
        ``None`` uses the backend default.
    magnitude_threshold : float
        Silence detection threshold — chunks with mean absolute magnitude
        below this value are skipped (default 0.02).
    scribe_vad : bool
        Enable Silero VAD for more accurate silence detection (default
        ``False``).
    scribe_backend : str
        Whisper/transcription backend (default ``"auto"``).
    slate_backend : str
        Text-translation backend: ``"googletrans"``, ``"libretranslate"``,
        ``"none"``, etc. (default ``"googletrans"``).
    voice_backend : str
        TTS backend: ``"gtts"`` or ``"piper"`` (default ``"gtts"``).
    voice_model : str or None
        Voice model name for TTS (Piper voice filename, etc.).
    voice_lang : str or None
        Explicit TTS language override.  ``None`` uses the detected/translated
        language.
    voice_match : bool
        Enable voice cloning from reference audio (default ``False``).
    lang_prefix : bool
        Prefix output text with the detected language name
        (e.g. ``"French: ..."``) (default ``False``).
    verbose : bool
        Print diagnostic messages (default ``False``).
    timers : bool
        Collect per-stage timing statistics (default ``False``).
    langswap_enabled : bool
        Enable bidirectional (LangSwap) translation mode (default ``False``).
    langswap_input_lang : str or None
        Primary input language for LangSwap.
    langswap_output_lang : str or None
        Primary output language for LangSwap.
    """

    # Language
    input_lang: Optional[str] = None
    output_lang: Optional[str] = None
    text_translation_target: Optional[str] = None

    # Audio / model
    model: Optional[str] = None
    magnitude_threshold: float = 0.02
    scribe_vad: bool = False

    # Backends
    scribe_backend: str = "auto"
    slate_backend: str = "googletrans"
    voice_backend: str = "gtts"
    voice_model: Optional[str] = None
    voice_lang: Optional[str] = None

    # Voice matching / cloning
    voice_match: bool = False
    lang_prefix: bool = False

    # Timing / debugging
    verbose: bool = False
    timers: bool = False

    # LangSwap (bidirectional translation)
    langswap_enabled: bool = False
    langswap_input_lang: Optional[str] = None
    langswap_output_lang: Optional[str] = None


# ---------------------------------------------------------------------------
# Output (file / audio sink) configuration
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    """File paths for text and audio outputs.

    All fields are optional.  ``None`` disables the corresponding output.

    Parameters
    ----------
    scribe_text_file : str or None
        Path to write Stage 1 (scribe / English) transcriptions.
    slate_text_file : str or None
        Path to write Stage 2 (slate / translated) text.
    output_audio_path : str or None
        Path to save scribe (English) TTS audio.
    slate_audio_path : str or None
        Path to save slate (translated) TTS audio.
    capture_voice_path : str or None
        Path to save captured input voice audio.
    """

    scribe_text_file: Optional[str] = None
    slate_text_file: Optional[str] = None
    output_audio_path: Optional[str] = None
    slate_audio_path: Optional[str] = None
    capture_voice_path: Optional[str] = None
