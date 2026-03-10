import argparse
import os
import subprocess
import sys


from .certs import generate_self_signed_cert
from .whisper_backend import download_whisper_cpp_model, _derive_whispercpp_model_name
def _set_default_env_vars():
    """
    Set default environment variables for whisper backends if not already set.
    Applies different defaults based on the operating system (Linux vs Windows).
    """
    import platform
    is_windows = platform.system() == "Windows"
    home = os.path.expanduser("~")
    
    # Set default WHISPERCPP_BIN if not already set
    if "WHISPERCPP_BIN" not in os.environ:
        if is_windows:
            default_bin = os.path.join(home, "whisper.cpp", "build", "bin", "Release", "whisper-cli.exe")
        else:
            default_bin = os.path.join(home, "whisper.cpp", "build", "bin", "whisper-cli")
        os.environ["WHISPERCPP_BIN"] = default_bin
    
    # Set default WHISPERCPP_MODEL_NAME if not already set
    if "WHISPERCPP_MODEL_NAME" not in os.environ:
        os.environ["WHISPERCPP_MODEL_NAME"] = "medium"
    
    # Set default WHISPERCPP_MODEL_DIR if not already set (relative to voicetran directory)
    if "WHISPERCPP_MODEL_DIR" not in os.environ:
        os.environ["WHISPERCPP_MODEL_DIR"] = "./models"
    
    # Set default WHISPERCPP_THREADS if not already set
    if "WHISPERCPP_THREADS" not in os.environ:
        os.environ["WHISPERCPP_THREADS"] = "4"
    
    # Set default WHISPER_CTRANSLATE2_DEVICE_INDEX if not already set
    if "WHISPER_CTRANSLATE2_DEVICE_INDEX" not in os.environ:
        os.environ["WHISPER_CTRANSLATE2_DEVICE_INDEX"] = "0"
    
    # Set default WHISPER_CTRANSLATE2_COMPUTE_TYPE if not already set
    if "WHISPER_CTRANSLATE2_COMPUTE_TYPE" not in os.environ:
        os.environ["WHISPER_CTRANSLATE2_COMPUTE_TYPE"] = "default"


from .config import (
    set_whisper_backend,
    set_whisper_cpp_config,
    set_whispercpp_cli_detect_lang,
    set_whispercpp_force_cli,
    set_whisper_ctranslate2_config,
)
from .stream_output import list_wasapi_loopback_devices
from .text_translator import set_translation_backend, set_libretranslate_config, set_translategemma_config, set_metanllb_config, set_marianmt_config
from .utils import resolve_path_with_fallback
from .vad import SILERO_AVAILABLE
from .pipelines import build_pipeline_config, execute_pipeline


from .config_options import (
    _detect_format,
    _dict_to_toml,
    _get_default_config,
    _load_config_file,
    _toml_load,
    _write_config_file,
)


def main():
    # Set default environment variables at startup
    _set_default_env_vars()

    # --- Pre-parse: handle --config and --genconfig before the full parse ---
    # This allows --genconfig to work without requiring an input source, and
    # allows --config to inject defaults before the main parser runs.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_parser.add_argument("--genconfig", nargs="?", const="anytran.json", default=None)
    pre_parser.add_argument("--voice-table-gen", action="store_true", default=False)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Real-time Audio Translator with 3-Stage Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  Stage 1 (Transcription): Voice/text -> English text
  Stage 2 (Translation):   English text -> output-lang text (if output-lang != en)
  Stage 3 (TTS):          Text -> Voice (if voice output requested)

Examples:
  # Transcribe YouTube video to English text
  anytran --youtube-url URL --youtube-api-key KEY --scribe-text output.txt
  
  # Transcribe and translate to French
  anytran --rtsp rtsp://stream --output-lang fr --slate-text french.txt
  
  # With voice output
  anytran --web --output-lang es --slate-voice
"""
    )
    
    # Input source (mutually exclusive)
    # required=False here because --config may supply the input source via set_defaults();
    # we enforce the requirement manually after parsing (see below).
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input", type=str, help="Input file (txt, mp3, mp4, wav)")
    input_group.add_argument("--rtsp", type=str, action="append", help="RTSP stream URL (can be repeated)")
    input_group.add_argument("--from-output", action="store_true", help="Capture system audio output (Windows WASAPI)")
    input_group.add_argument("--web", action="store_true", help="Start web server for browser mic input")
    input_group.add_argument("--youtube-url", type=str, help="YouTube video URL")
    
    # Language configuration
    parser.add_argument("--input-lang", type=str, default=None, help="Input language code (default: auto). Required for text file input.")
    parser.add_argument("--output-lang", type=str, default="en", help="Output language code (default: en)")
    
    # Stage 1 outputs (English transcription)
    parser.add_argument("--scribe-text", type=str, help="Save English transcription to text file")
    parser.add_argument("--scribe-voice", type=str, help="Save English transcription as audio file")
    
    # Stage 2 outputs (Translation to output-lang, only if output-lang != en)
    parser.add_argument("--slate-text", type=str, help="Save translation to text file")
    parser.add_argument("--slate-voice", type=str, help="Save translated voice output as audio file")

    # Capture original input voice (all inputs except --input)
    parser.add_argument("--capture-voice", type=str, help="Save the original input audio to a file (not supported with --input)")
    
    # Scribe (Speech-to-Text) options
    scribe_group = parser.add_argument_group("scribe options (speech-to-text / Stage 1)")
    scribe_group.add_argument(
        "--scribe-backend",
        type=str,
        default="faster-whisper",
        choices=["whispercpp", "whispercpp-cli", "faster-whisper", "whisper-ctranslate2"],
        help="Whisper backend (default: faster-whisper). Use 'whispercpp-cli' for CLI-based inference."
    )
    scribe_group.add_argument("--scribe-model", type=str, default=os.environ.get("WHISPERCPP_MODEL_NAME", "medium"), help="Whisper model name or path (default: WHISPERCPP_MODEL_NAME env var or 'medium'). For whispercpp: model name like 'small', 'medium', or full path to .bin file. For other backends: model name or path.")
    scribe_group.add_argument("--scribe-vad", action="store_true", help="Use Silero VAD for speech detection")
    scribe_group.add_argument("--no-scribe-vad", action="store_false", dest="scribe_vad", default=False, help="Disable Silero VAD for speech detection (default)")
    scribe_group.add_argument("--magnitude-threshold", type=float, default=0.01, help="Silence detection threshold (default: 0.01)")
    
    # Whisper.cpp specific
    whispercpp_bin_default = os.environ.get("WHISPERCPP_BIN", None)
    whispercpp_model_dir_default = os.environ.get("WHISPERCPP_MODEL_DIR", "./models")
    scribe_group.add_argument("--whispercpp-bin", type=str, default=whispercpp_bin_default, help="Path to whisper.cpp binary (can also set WHISPERCPP_BIN env var)")
    scribe_group.add_argument("--whispercpp-model-dir", type=str, default=whispercpp_model_dir_default, help="Directory containing whisper.cpp models (can also set WHISPERCPP_MODEL_DIR env var, default: ./models)")
    scribe_group.add_argument("--whispercpp-threads", type=int, default=4, help="Threads for whisper.cpp (default: 4)")
    scribe_group.add_argument("--no-auto-download", action="store_true", help="Disable auto-download of missing whisper.cpp models (default: auto-download enabled)")
    scribe_group.add_argument("--auto-download", action="store_false", dest="no_auto_download", default=False, help="Enable auto-download of missing whisper.cpp models (default)")
    scribe_group.add_argument("--whispercpp-cli-detect-lang", action="store_true", help="Use whisper.cpp CLI for language detection")
    scribe_group.add_argument("--no-whispercpp-cli-detect-lang", action="store_false", dest="whispercpp_cli_detect_lang", default=False, help="Disable whisper.cpp CLI language detection (default)")
    
    # Whisper-ctranslate2 specific
    scribe_group.add_argument("--whisper-ctranslate2-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for whisper-ctranslate2 (default: auto)")
    scribe_group.add_argument("--whisper-ctranslate2-device-index", type=int, help="Device index for whisper-ctranslate2")
    scribe_group.add_argument("--whisper-ctranslate2-compute-type", type=str, default="default", help="Compute type for whisper-ctranslate2 (default)")
    
    # Slate (Text Translation) options
    slate_group = parser.add_argument_group("slate options (text translation / Stage 2)")
    slate_group.add_argument("--slate-backend", type=str, default="googletrans", choices=["googletrans", "libretranslate", "translategemma", "metanllb", "marianmt", "none"], help="Text translation backend (default: googletrans)")
    slate_group.add_argument("--slate-model", type=str, default=None, help="Translation model name (used with --slate-backend translategemma/metanllb/marianmt). Default depends on backend: translategemma=google/translategemma-12b-it, metanllb=facebook/nllb-200-1.3B, marianmt=auto-derived from language pair)")
    slate_group.add_argument("--libretranslate-url", type=str, help="LibreTranslate API URL")
    
    # Voice (TTS) options
    voice_group = parser.add_argument_group("voice options (text-to-speech / Stage 3)")
    voice_group.add_argument("--voice-backend", type=str, default="auto", choices=["piper", "gtts", "custom", "fish", "indextts", "coqui", "auto"], help="TTS backend. 'auto' (default) prefers Piper if available and falls back to gTTS otherwise. Use 'piper' to force Piper TTS, 'gtts' to force Google TTS, 'custom' to use Qwen3-TTS CustomVoice/Base models, 'fish' to use fish-speech (supports fishaudio/s1-mini and fishaudio/fish-speech-1.5 models), 'indextts' to use IndexTTS (supports IndexTeam/IndexTTS-2 and compatible models; can optionally use --voice-match to supply a speaker prompt; install with: GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/index-tts/index-tts.git && pip install 'anytran[index-tts]'), or 'coqui' to use coqui-tts XTTS v2 (Python 3.12-compatible, supports 17 languages and zero-shot voice cloning via --voice-match; install with: pip install 'anytran[coqui]').")
    voice_group.add_argument("--voice-model", type=str, default="en_US-lessac-medium", help="Voice model name for TTS (default: en_US-lessac-medium). Used as the Piper voice model when --voice-backend piper is selected.")
    voice_group.add_argument("--voice-lang", type=str, help="Override TTS language")
    voice_group.add_argument("--voice-match", action="store_true", help="Auto-select Piper voice based on input voice characteristics")
    voice_group.add_argument("--no-voice-match", action="store_false", dest="voice_match", default=False, help="Disable auto-selection of Piper voice (default)")
    
    # Audio processing
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window length in seconds (default: 5.0)")
    parser.add_argument("--overlap-seconds", type=float, default=0.0, help="Overlap length in seconds (default: 0.0)")
    
    # MQTT publishing
    parser.add_argument("--mqtt-broker", type=str, help="MQTT broker hostname/IP")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT port (default: 1883)")
    parser.add_argument("--mqtt-username", type=str, help="MQTT username")
    parser.add_argument("--mqtt-password", type=str, help="MQTT password")
    parser.add_argument("--mqtt-topic", type=str, default="translation", help="MQTT topic (default: translation)")
    parser.add_argument("--mqtt-topic-names", type=str, action="append", help="Custom MQTT topic for each RTSP stream")
    
    # Web server options
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=8443, help="Web server port (default: 8443)")
    parser.add_argument("--web-ssl-cert", type=str, help="SSL certificate file for HTTPS")
    parser.add_argument("--web-ssl-key", type=str, help="SSL private key file for HTTPS")
    parser.add_argument("--web-ssl-self-signed", action="store_true", help="Generate self-signed SSL cert")
    parser.add_argument("--no-web-ssl-self-signed", action="store_false", dest="web_ssl_self_signed", default=False, help="Disable self-signed SSL cert generation (default)")
    
    # YouTube specific
    parser.add_argument("--youtube-api-key", type=str, help="YouTube Data API v3 key")
    parser.add_argument("--youtube-js-runtime", type=str, help="yt-dlp JS runtime (default: node)")
    parser.add_argument("--youtube-remote-components", type=str, default="ejs:github", help="yt-dlp remote components")
    
    # System output capture
    parser.add_argument("--output-device", type=str, help="Output device name for loopback capture (Windows)")
    parser.add_argument("--list-output-devices", action="store_true", help="List available output devices (Windows)")
    
    # Chat logging
    parser.add_argument("--chat-log", type=str, default="./chat", help="Chat log directory (default: ./chat)")
    
    # Misc
    parser.add_argument(
        "--batch-input-text",
        type=int,
        default=0,
        metavar="N",
        help="Batch N lines/sentences together for text translation when input is a text file (default: 0, no batching)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-verbose", action="store_false", dest="verbose", default=False, help="Disable verbose output (default)")
    parser.add_argument(
        "--timers",
        action="store_true",
        help=(
            "Print timing summary aggregated by stage only (no overall or overhead breakdown). "
            "If used together with --timers-all, this option is superseded by --timers-all."
        ),
    )
    parser.add_argument(
        "--no-timers",
        action="store_false",
        dest="timers",
        default=False,
        help="Disable timing summary (default)",
    )
    parser.add_argument(
        "--timers-all",
        action="store_true",
        help=(
            "Print all timing summaries: overall, by stage, and translation-overhead breakdown. "
            "This supersedes --timers when both options are provided."
        ),
    )
    parser.add_argument(
        "--no-timers-all",
        action="store_false",
        dest="timers_all",
        default=False,
        help="Disable full timing summaries (default)",
    )
    parser.add_argument("--lang-prefix", action="store_true", default=False, help="Add language prefix (e.g. 'English: ', 'French: ') to each output text line (default: False)")
    parser.add_argument("--no-lang-prefix", action="store_false", dest="lang_prefix", default=False, help="Disable language prefix in output (default)")

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not remove temporary files (for debugging)",
    )
    parser.add_argument(
        "--no-keep-temp",
        action="store_false",
        dest="keep_temp",
        default=False,
        help="Remove temporary files after use (default)",
    )

    parser.add_argument(
        "--dedup",
        action="store_true",
        default=False,
        help="Enable deduplication of text output (default: off)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_false",
        dest="dedup",
        default=False,
        help="Disable deduplication of text output (default)",
    )

    parser.add_argument(
            "--looptran",
            type=int,
            default=0,
            metavar="N",
            help=(
                "Repeat translation N additional times, swapping input/output languages each time "
                "(text file input with --slate-text and different --input-lang/--output-lang required). "
                "Each iteration uses the previous --slate-text output as input and creates a new "
                "--slate-text file with a '_<i>' postfix (default: 0)."
            ),
    )

    parser.add_argument(
            "--tran-converge",
            nargs="?",
            const=0,
            type=int,
            default=None,
            help=(
                "When used with --looptran, detect convergence by comparing each output file with the "
                "output from two iterations back (same language). If the number of differing lines is "
                "less than or equal to the specified threshold, report convergence and stop looping early. "
                "Omitting this flag disables convergence checking (default: None). Providing '--tran-converge' "
                "with no value enables exact-match checking (threshold 0). Providing '--tran-converge N' sets "
                "the allowed number of differing lines to N."
            ),
    )

    parser.add_argument(
        "--no-norm",
        action="store_true",
        default=False,
        help="Disable text normalization before writing output to file (normalization is enabled by default)",
    )
    parser.add_argument(
        "--norm",
        action="store_false",
        dest="no_norm",
        default=False,
        help="Enable text normalization before writing output to file (default)",
    )
    parser.add_argument(
        "--no-input-norm",
        action="store_true",
        default=False,
        help=(
            "Disable normalization of the original input text file before processing when using "
            "--input with a text file. This keeps the input exactly as-is; normalization is "
            "enabled by default and is separate from output text normalization controlled by "
            "--no-norm."
        ),
    )
    parser.add_argument(
        "--input-norm",
        action="store_false",
        dest="no_input_norm",
        default=False,
        help="Enable normalization of the input text file before processing (default)",
    )

    # Voice table generation options
    voice_table_group = parser.add_argument_group("voice table generation options")
    voice_table_group.add_argument(
        "--voice-table-gen",
        action="store_true",
        default=False,
        help="Generate or update the voice table JSON file with voice features and exit.",
    )
    voice_table_group.add_argument(
        "--voice-table-lang",
        type=str,
        default="fr",
        metavar="LANGUAGES",
        help="Comma-separated language codes for voice table generation (default: fr). Use 'all' for all languages.",
    )
    voice_table_group.add_argument(
        "--voice-table-output",
        type=str,
        default="src/anytran/voice_table.json",
        metavar="PATH",
        help="Output JSON file path for voice table generation (default: src/anytran/voice_table.json).",
    )

    # Config file options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Load settings from a JSON config file. CLI options override values in the config file.",
    )
    parser.add_argument(
        "--genconfig",
        nargs="?",
        const="anytran.json",
        default=None,
        metavar="PATH",
        help=(
            "Generate a JSON config file with all default settings and exit. "
            "Optionally provide a file path (default: anytran.json). "
            "Use '-' to print to stdout."
        ),
    )

    # Capture original argparse defaults BEFORE set_defaults() mutates them,
    # so that _print_non_default_args can report config-file values as non-default.
    original_defaults = {action.dest: action.default for action in parser._actions}

    # Apply config file defaults before parsing (CLI args override config values)
    if pre_args.config:
        file_defaults = _load_config_file(pre_args.config)
        parser.set_defaults(**file_defaults)

    args = parser.parse_args()

    # --voice-table-gen: generate/update the voice table and exit.
    # Must happen before the input-source enforcement check.
    if args.voice_table_gen:
        from .voice_table import get_selected_languages, run as run_voice_table
        run_voice_table(get_selected_languages(args.voice_table_lang), args.voice_table_output)
        return 0

    # --genconfig: write the actual parsed values (not static defaults) and exit.
    # Must happen before the input-source enforcement check so that --genconfig
    # works without requiring an input source.
    # --config and --genconfig may be combined: the config file values (plus any
    # CLI overrides) are already reflected in args via set_defaults above.
    if args.genconfig is not None:
        _print_non_default_args(args, original_defaults)
        cfg = {k: v for k, v in vars(args).items() if k not in ("config", "genconfig")}
        _write_config_file(cfg, args.genconfig)
        return 0

    # Enforce that exactly one input source was provided (via CLI or config file).
    if not any([args.input, args.rtsp, args.from_output, args.web, args.youtube_url]):
        parser.error(
            "one of the arguments --input --rtsp --from-output --web --youtube-url is required"
        )

    # Initialize KEEP_TEMP global for all modules
    import builtins
    builtins.KEEP_TEMP = getattr(args, "keep_temp", False)

    if args.list_output_devices:
        list_wasapi_loopback_devices()
        return 0

    # Validate pipeline logic
    _validate_pipeline_args(args, parser)
    
    # Configure backends
    _configure_backends(args)
    
    # Auto-create chat-log directory
    if args.chat_log:
        _ensure_chat_log_dir(args)
    
    # Generate self-signed cert if needed
    if args.web and args.web_ssl_self_signed:
        _generate_ssl_cert_if_needed(args)
    
    # Validate window/overlap settings
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be greater than 0")
    if args.overlap_seconds < 0:
        parser.error("--overlap-seconds must be 0 or greater")
    if args.overlap_seconds >= args.window_seconds:
        parser.error("--overlap-seconds must be less than --window-seconds")
    
    # When a config file was supplied, print non-default settings (config file
    # values + any CLI overrides) before the pipeline starts.
    if args.config:
        _print_non_default_args(args, original_defaults)

    # Map new arguments to pipeline configuration
    pipeline_config = build_pipeline_config(args)
    
    # Execute pipeline based on input source
    return execute_pipeline(args, pipeline_config)


def _print_non_default_args(args, original_defaults):
    """Print all argument values that differ from the original argparse defaults.

    ``original_defaults`` must be the ``{dest: default}`` dict captured from
    the parser *before* any ``set_defaults()`` call, so that config-file values
    are correctly reported as non-default.

    The meta-keys ``config`` and ``genconfig`` are omitted from the output.
    """
    non_defaults = {
        k: v
        for k, v in vars(args).items()
        if k not in ("config", "genconfig") and v != original_defaults.get(k)
    }
    if non_defaults:
        print("Non-default settings:")
        for k in sorted(non_defaults):
            flag = "--" + k.replace("_", "-")
            print(f"  {flag} = {non_defaults[k]!r}")


def _validate_pipeline_args(args, parser):
    """Validate argument combinations and pipeline logic."""
    
    # Input validation
    if args.input and not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")
    
    # Check if input is text file (Stage 1 skips voice transcription)
    if args.input and args.input.endswith('.txt'):
        # Text input file - must have input_lang for translation
        if not args.input_lang or args.input_lang.lower() == "auto":
            parser.error("--input-lang must be specified (not 'auto') when using text file input")
    
    # Default input_lang to None (auto) if not specified
    if not args.input_lang:
        args.input_lang = None
    
    # Streaming sources need at least one output
    if (args.rtsp or args.from_output or args.youtube_url or args.web):
        if not (args.scribe_text or args.scribe_voice or args.slate_text or args.slate_voice or args.mqtt_broker):
            if args.youtube_url or args.from_output:
                parser.error(f"Streaming source requires at least one output: --scribe-text, --slate-text, --mqtt-broker, etc.")
    
    # YouTube validation
    if args.youtube_url and not args.youtube_api_key:
        parser.error("--youtube-url requires --youtube-api-key")
    
    # Web SSL validation
    if args.web and (bool(args.web_ssl_cert) ^ bool(args.web_ssl_key)):
        parser.error("--web-ssl-cert and --web-ssl-key must be provided together")
    
    # MQTT topic names must match RTSP count
    if args.rtsp and args.mqtt_topic_names and len(args.mqtt_topic_names) != len(args.rtsp):
        parser.error(f"Number of --mqtt-topic-names ({len(args.mqtt_topic_names)}) must match number of --rtsp streams ({len(args.rtsp)})")
    
    # VAD check
    if args.scribe_vad and not SILERO_AVAILABLE:
        print("Warning: --scribe-vad specified but Silero VAD not installed. Install with: pip install silero-vad")
        print("Falling back to magnitude threshold for speech detection.")
    
    # Piper check
    if args.voice_backend == "piper":
        try:
            result = subprocess.run(["piper", "--help"], capture_output=True, timeout=2)
            if result.returncode != 0:
                print("Warning: --voice-backend piper specified but Piper not found.")
                print("Falling back to gTTS.")
                args.voice_backend = "gtts"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Warning: --voice-backend piper specified but Piper not found.")
            print("Falling back to gTTS.")
            args.voice_backend = "gtts"


def _configure_backends(args):
    """Configure whisper and translation backends."""
    # Check if input is a text file - if so, skip Whisper configuration
    is_text_input = args.input and args.input.endswith('.txt')
    
    # Configure text translation backend (always needed)
    if args.slate_backend:
        set_translation_backend(args.slate_backend)
        if args.slate_backend == "libretranslate" and args.libretranslate_url:
            set_libretranslate_config(args.libretranslate_url)
        elif args.slate_backend == "translategemma":
            set_translategemma_config(args.slate_model or "google/translategemma-12b-it")
        elif args.slate_backend == "metanllb":
            set_metanllb_config(args.slate_model or "facebook/nllb-200-1.3B")
        elif args.slate_backend == "marianmt":
            set_marianmt_config(args.slate_model)
    
    # Skip Whisper backend configuration for text file inputs
    if is_text_input:
        return
    
    # Configure Whisper backend for audio processing
    # Handle whispercpp-cli backend by setting force_cli flag and using whispercpp backend
    if args.scribe_backend == "whispercpp-cli":
        set_whisper_backend("whispercpp")
        set_whispercpp_force_cli(True)
    else:
        set_whisper_backend(args.scribe_backend)
    
    set_whispercpp_cli_detect_lang(args.whispercpp_cli_detect_lang)
    
    # Configure whisper.cpp
    if args.scribe_backend == "whispercpp" or args.scribe_backend == "whispercpp-cli":
        _configure_whispercpp(args)
    
    # Configure whisper-ctranslate2
    if args.scribe_backend == "whisper-ctranslate2":
        _configure_whisper_ctranslate2(args)


def _configure_whispercpp(args):
    """Configure whisper.cpp backend."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cleanup_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    repo_root = os.path.abspath(os.path.join(cleanup_root, ".."))

    # Determine models directory (uses default from argument parser which includes env var)
    models_dir = args.whispercpp_model_dir
    
    # Expand to absolute path if relative
    if not os.path.isabs(models_dir):
        # Try relative to cleanup_root first, then repo_root
        candidate_path = os.path.join(cleanup_root, models_dir)
        if os.path.exists(candidate_path):
            models_dir = candidate_path
        else:
            candidate_path = os.path.join(repo_root, models_dir)
            if os.path.exists(candidate_path):
                models_dir = candidate_path
            else:
                # Create it relative to cleanup_root
                models_dir = os.path.join(cleanup_root, models_dir)

    model_path = None
    
    # Get model name from --scribe-model option (which already has the default from env or "medium")
    requested_model_name = args.scribe_model

    if requested_model_name:
        # Check if it's an absolute path or existing file first
        candidate_path = os.path.expanduser(requested_model_name)
        if os.path.exists(candidate_path):
            model_path = candidate_path
            if args.verbose:
                print(f"Using model file: {model_path}")
        else:
            # Construct path from model name
            candidate_path = os.path.join(models_dir, f"ggml-{requested_model_name}.bin")
            if os.path.exists(candidate_path):
                model_path = candidate_path
            elif not args.no_auto_download:
                # Auto-download is enabled by default
                downloaded = download_whisper_cpp_model(requested_model_name, models_dir, verbose=True)
                if downloaded:
                    model_path = downloaded
                    print(f"Downloaded model: {requested_model_name}")
                    print(f"Source: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{requested_model_name}.bin")
                    print(f"Target: {model_path}")
            else:
                print(f"Error: whispercpp model '{requested_model_name}' not found. Enable auto-download or specify model path.")
                sys.exit(1)

    bin_path = resolve_path_with_fallback(args.whispercpp_bin, repo_root)
    model_path = resolve_path_with_fallback(model_path, repo_root)

    set_whisper_cpp_config(
        bin_path=bin_path,
        model_path=model_path,
        threads=args.whispercpp_threads,
    )

    if args.verbose:
        print("Backend: whispercpp")
        print(f"whispercpp model: {model_path}")
        print(f"whispercpp threads: {args.whispercpp_threads}")

    if not model_path or not os.path.exists(model_path):
        print(f"Error: whispercpp model file not found: {model_path}")
        if args.no_auto_download:
            print("Auto-download is disabled. Enable auto-download to download missing models.")
        sys.exit(1)


def _configure_whisper_ctranslate2(args):
    """Configure whisper-ctranslate2 backend."""
    model_name = args.scribe_model
    if model_name:
        resolved_model_path = os.path.expanduser(model_name)
        if os.path.exists(resolved_model_path):
            model_name = resolved_model_path
        elif any(sep in model_name for sep in ("/", "\\")) or model_name.startswith((".", "~")):
            print(f"Error: whisper-ctranslate2 model path not found: {resolved_model_path}")
            sys.exit(1)

    set_whisper_ctranslate2_config(
        model_name=model_name,
        device=args.whisper_ctranslate2_device,
        device_index=args.whisper_ctranslate2_device_index,
        compute_type=args.whisper_ctranslate2_compute_type,
    )
    
    if args.verbose:
        print("Backend: whisper-ctranslate2")
        print(f"whisper-ctranslate2 model: {model_name}")
        print(f"whisper-ctranslate2 device: {args.whisper_ctranslate2_device}")
        if args.whisper_ctranslate2_device_index is not None:
            print(f"whisper-ctranslate2 device index: {args.whisper_ctranslate2_device_index}")
        print(f"whisper-ctranslate2 compute type: {args.whisper_ctranslate2_compute_type}")


def _ensure_chat_log_dir(args):
    """Ensure chat log directory exists."""
    chat_log_dir = os.path.expanduser(args.chat_log)
    created = False
    try:
        if not os.path.isdir(chat_log_dir):
            os.makedirs(chat_log_dir, exist_ok=True)
            created = True
    except OSError as exc:
        print(f"Error: Unable to create chat-log directory '{chat_log_dir}': {exc}")
        sys.exit(1)
    if created and args.verbose:
        print(f"Created chat-log directory: {chat_log_dir}")
    args.chat_log = chat_log_dir


def _generate_ssl_cert_if_needed(args):
    """Generate self-signed SSL certificate if needed."""
    if not (args.web_ssl_cert and args.web_ssl_key):
        args.web_ssl_cert = args.web_ssl_cert or os.path.join(os.getcwd(), "selfsigned.crt")
        args.web_ssl_key = args.web_ssl_key or os.path.join(os.getcwd(), "selfsigned.key")
        if not (os.path.exists(args.web_ssl_cert) and os.path.exists(args.web_ssl_key)):
            try:
                generate_self_signed_cert(
                    args.web_ssl_cert,
                    args.web_ssl_key,
                    common_name=args.web_host,
                    verbose=args.verbose,
                )
                print(f"Generated self-signed cert: {args.web_ssl_cert}")
                print(f"Generated self-signed key: {args.web_ssl_key}")
            except Exception as exc:
                print(f"Error generating self-signed cert: {exc}")
                sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
