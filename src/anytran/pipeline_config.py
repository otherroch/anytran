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
* **RunnerConfig** - Top-level bundle combining PipelineConfig, OutputConfig,
  and MQTTConfig.  Runner functions accept this as a single argument instead
  of ~30 individual keyword parameters.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


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


# ---------------------------------------------------------------------------
# Stream context (per-stream mutable state)
# ---------------------------------------------------------------------------

@dataclass
class StreamContext:
    """Mutable state that lives for the lifetime of one audio stream.

    This groups the parameters that change per-stream but are constant per-
    *call* to ``process_audio_chunk``, allowing us to drop six individual
    keyword arguments in favour of a single ``ctx`` parameter.

    Parameters
    ----------
    stream_id : str, int or None
        Human-readable identifier for the stream (logging only).
    timing_stats : object or None
        Accumulator for timing statistics (``TimingsAggregator`` instance).
    scribe_tts_segments : list or None
        Collected PCM segments for scribe (English) TTS audio.
    slate_tts_segments : list or None
        Collected PCM segments for slate (translated) TTS audio.
    chat_logger : object or None
        ChatLogger instance (RTSP chat logging).
    rtsp_ip : str or None
        IP address extracted from RTSP URL (RTSP chat logging).
    """

    stream_id: Optional[str] = None
    timing_stats: Optional[object] = None
    scribe_tts_segments: Optional[List] = None
    slate_tts_segments: Optional[List] = None
    chat_logger: Optional[object] = None
    rtsp_ip: Optional[str] = None


# ---------------------------------------------------------------------------
# Runner configuration (top-level bundle)
# ---------------------------------------------------------------------------

@dataclass
class RunnerConfig:
    """Top-level configuration bundle for runner functions.

    Combines ``PipelineConfig``, ``OutputConfig``, and ``MQTTConfig`` into
    a single parameter that replaces the ~30 individual keyword arguments
    previously passed to each runner.

    Parameters
    ----------
    pipeline : PipelineConfig
        Pipeline-wide behavioural settings.
    output : OutputConfig
        File paths for audio and text outputs.
    mqtt : MQTTConfig
        MQTT broker connection and topic.
    **extra : Any
        Runner-specific parameters that don't fit the common pattern
        (e.g. ``youtube_js_runtime``, ``output_device``, ``host``, ``port``).
    """

    pipeline: PipelineConfig
    output: OutputConfig
    mqtt: MQTTConfig
    extra: dict = field(default_factory=dict)

    # -- convenience forwards ------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a key in ``extra``."""
        return self.extra.get(key, default)

    @property
    def windows(self):
        """Number of processing windows (convenience forward)."""
        return self.pipeline.window_seconds

    @property
    def overlap(self):
        """Overlap in seconds (convenience forward)."""
        return self.pipeline.overlap_seconds

    # -- backward-compat factory ----------------------------------------------

    @classmethod
    def _from_kwargs(cls, **kwargs) -> "RunnerConfig":
        """Construct a RunnerConfig from old-style individual keyword args.

        This factory recognises the legacy kwarg names used in the test suite
        and CLI entry points and groups them into the appropriate sub-configs.

        Recognised legacy keys
        ----------------------
        Pipeline (behaviour):
            input_lang, output_lang, text_translation_target, model,
            magnitude_threshold, scribe_vad, window_seconds, overlap_seconds,
            scribe_backend, slate_backend, voice_backend, voice_model,
            voice_lang, voice_match, lang_prefix, normalize, dedup,
            slate_no_opt, batch, verbose, timers, timers_all, keep_temp,
            langswap_enabled, langswap_input_lang, langswap_output_lang

        Output (file paths):
            scribe_text_file, slate_text_file, output_audio_path,
            slate_audio_path, capture_voice_path, chat_log_dir

        MQTT:
            mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic

        Runner-specific extras:
            Everything else is collected into ``extra``.
        """

        # -- Pipeline kwargs --
        pipeline_fields = {
            "input_lang", "output_lang", "text_translation_target",
            "model", "magnitude_threshold", "scribe_vad",
            "window_seconds", "overlap_seconds",
            "scribe_backend", "slate_backend", "voice_backend",
            "voice_model", "voice_lang",
            "voice_match", "lang_prefix",
            "normalize", "dedup", "slate_no_opt", "batch",
            "verbose", "timers", "timers_all",
            "keep_temp",
            "langswap_enabled", "langswap_input_lang", "langswap_output_lang",
        }
        pipeline_kw = {k: v for k, v in kwargs.items() if k in pipeline_fields and v is not None}

        # -- Output kwargs --
        output_fields = {
            "scribe_text_file", "slate_text_file", "output_audio_path",
            "slate_audio_path", "capture_voice_path", "chat_log_dir",
        }
        output_kw = {k: v for k, v in kwargs.items() if k in output_fields and v is not None}

        # -- MQTT kwargs --
        mqtt_fields = {
            "mqtt_broker", "mqtt_port", "mqtt_username",
            "mqtt_password", "mqtt_topic",
        }
        mqtt_kw = {}
        mqtt_field_map = {
            "mqtt_broker": "broker",
            "mqtt_port": "port",
            "mqtt_username": "username",
            "mqtt_password": "password",
            "mqtt_topic": "topic",
        }
        for legacy_key, canon_key in mqtt_field_map.items():
            if legacy_key in kwargs and kwargs[legacy_key] is not None:
                mqtt_kw[canon_key] = kwargs[legacy_key]

        # -- Extras --
        extra_keys = pipeline_fields | output_fields | mqtt_fields
        extra_kw = {k: v for k, v in kwargs.items() if k not in extra_keys}

        return cls(
            pipeline=PipelineConfig(**pipeline_kw),
            output=OutputConfig(**output_kw) if output_kw else OutputConfig(),
            mqtt=MQTTConfig(**mqtt_kw) if mqtt_kw else MQTTConfig(),
            extra=extra_kw if extra_kw else {},
        )
