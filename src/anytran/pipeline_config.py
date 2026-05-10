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
* **OutputConfig** - File paths for audio and text outputs.
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
    password : str or None
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
        return self.broker is not None


# ---------------------------------------------------------------------------
# Pipeline (behaviour) configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Pipeline-wide settings that rarely change during a run.

    This consolidates the ~25+ scalar parameters that were previously passed
    individually through every layer of the call graph.

    Parameters
    ----------
    # Language
    input_lang : str or None
        Source language code (e.g. ``"en"``, ``"de"``, ``"auto"``).
    output_lang : str or None
        Target language for transcription.
    text_translation_target : str or None
        Target language for Stage 2 text translation.

    # Audio / model
    model : str or None
        Model name (default ``"medium"`` for whispercpp).
    magnitude_threshold : float
        Silence detection threshold (default 0.02).
    scribe_vad : bool
        Enable Silero VAD (default ``False``).

    # Windows / overlap
    window_seconds : float
        Processing window size in seconds (default 5.0).
    overlap_seconds : float
        Overlap between consecutive windows (default 0.0).

    # Backends
    scribe_backend : str
        Whisper/transcription backend (default ``"auto"``).
    slate_backend : str
        Translation backend (default ``"googletrans"``).
    voice_backend : str
        TTS backend (default ``"gtts"``).
    voice_model : str or None
        Voice model for TTS.
    voice_lang : str or None
        Override TTS language.

    # Voice matching / cloning
    voice_match : bool
        Enable voice cloning (default ``False``).
    lang_prefix : bool
        Prefix output text with detected language name (default ``False``).

    # Text processing
    normalize : bool
        Normalize output text (default ``True``).
    dedup : bool
        Deduplicate consecutive outputs (default ``False``).
    slate_no_opt : bool
        Disable direct-translation optimisation (default ``False``).
    batch : int
        Batch size for sentence translation (0 = no batching).

    # Timing / debugging
    verbose : bool
        Print diagnostic messages (default ``False``).
    timers : bool
        Collect per-stage timing statistics (default ``False``).
    timers_all : bool
        Collect *all* timing categories (default ``False``).

    # File input specific
    keep_temp : bool
        Keep temporary files (default ``False``).

    # Chat logging (RTSP)
    chat_logger : object or None
        ChatLogger instance for RTSP chat logging.
    rtsp_ip : str or None
        IP address extracted from RTSP URL.

    # LangSwap (bidirectional translation)
    langswap_enabled : bool
    langswap_input_lang : str or None
    langswap_output_lang : str or None
    """

    # Language
    input_lang: Optional[str] = None
    output_lang: Optional[str] = None
    text_translation_target: Optional[str] = None

    # Audio / model
    model: Optional[str] = None
    magnitude_threshold: float = 0.02
    scribe_vad: bool = False

    # Windows / overlap
    window_seconds: float = 5.0
    overlap_seconds: float = 0.0

    # Backends
    scribe_backend: str = "auto"
    slate_backend: str = "googletrans"
    voice_backend: str = "gtts"
    voice_model: Optional[str] = None
    voice_lang: Optional[str] = None

    # Voice matching / cloning
    voice_match: bool = False
    lang_prefix: bool = False

    # Text processing
    normalize: bool = True
    dedup: bool = False
    slate_no_opt: bool = False
    batch: int = 0

    # Timing / debugging
    verbose: bool = False
    timers: bool = False
    timers_all: bool = False

    # File input specific
    keep_temp: bool = False

    # Chat logging (RTSP)
    chat_logger: Optional[object] = None
    rtsp_ip: Optional[str] = None

    # LangSwap
    langswap_enabled: bool = False
    langswap_input_lang: Optional[str] = None
    langswap_output_lang: Optional[str] = None


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    """File paths for audio and text outputs.

    Parameters
    ----------
    output_audio_path : str or None
        Path to save scribe (English) audio.
    slate_audio_path : str or None
        Path to save slate (translated) audio.
    scribe_text_file : str or None
        Path to save scribe (English) text.
    slate_text_file : str or None
        Path to save slate (translated) text.
    capture_voice_path : str or None
        Path to save captured input voice audio.
    chat_log_dir : str or None
        Directory for chat log files (RTSP).
    """

    output_audio_path: Optional[str] = None
    slate_audio_path: Optional[str] = None
    scribe_text_file: Optional[str] = None
    slate_text_file: Optional[str] = None
    capture_voice_path: Optional[str] = None
    chat_log_dir: Optional[str] = None