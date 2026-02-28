"""Config file utilities for the anytran CLI.

Provides helpers to generate, read, and write configuration files in JSON
and TOML formats.  These functions are used by ``main.py`` to implement
the ``--config`` and ``--genconfig`` command-line options.
"""
import json
import os
import sys


def _get_default_config():
    """Return a dict of all CLI option defaults suitable for use as a config file."""
    import platform
    is_windows = platform.system() == "Windows"
    home = os.path.expanduser("~")
    if is_windows:
        default_bin = os.path.join(home, "whisper.cpp", "build", "bin", "Release", "whisper-cli.exe")
    else:
        default_bin = os.path.join(home, "whisper.cpp", "build", "bin", "whisper-cli")
    return {
        # Input source (set at most one; all default to None/False)
        "input": None,
        "rtsp": None,
        "from_output": False,
        "web": False,
        "youtube_url": None,
        # Language
        "input_lang": None,
        "output_lang": "en",
        # Stage outputs
        "scribe_text": None,
        "scribe_voice": None,
        "slate_text": None,
        "slate_voice": None,
        # Scribe / speech-to-text
        "scribe_backend": "faster-whisper",
        "scribe_model": os.environ.get("WHISPERCPP_MODEL_NAME", "medium"),
        "scribe_vad": False,
        "magnitude_threshold": 0.01,
        # whisper.cpp
        "whispercpp_bin": default_bin,
        "whispercpp_model_dir": os.environ.get("WHISPERCPP_MODEL_DIR", "./models"),
        "whispercpp_threads": 4,
        "no_auto_download": False,
        "whispercpp_cli_detect_lang": False,
        # whisper-ctranslate2
        "whisper_ctranslate2_device": "auto",
        "whisper_ctranslate2_device_index": None,
        "whisper_ctranslate2_compute_type": "default",
        # Slate / text translation
        "slate_backend": "googletrans",
        "slate_model": None,
        "libretranslate_url": None,
        # Voice / TTS
        "voice_backend": "gtts",
        "voice_model": "en_US-lessac-medium",
        "voice_lang": None,
        "voice_match": False,
        # Audio processing
        "window_seconds": 5.0,
        "overlap_seconds": 0.0,
        # MQTT
        "mqtt_broker": None,
        "mqtt_port": 1883,
        "mqtt_username": None,
        "mqtt_password": None,
        "mqtt_topic": "translation",
        "mqtt_topic_names": None,
        # Web server
        "web_host": "0.0.0.0",
        "web_port": 8443,
        "web_ssl_cert": None,
        "web_ssl_key": None,
        "web_ssl_self_signed": False,
        # YouTube
        "youtube_api_key": None,
        "youtube_js_runtime": None,
        "youtube_remote_components": "ejs:github",
        # System output
        "output_device": None,
        "list_output_devices": False,
        # Chat log
        "chat_log": "./chat",
        # Misc
        "batch_input_text": 0,
        "verbose": False,
        "timers": False,
        "timers_all": False,
        "lang_prefix": False,
        "keep_temp": False,
        "dedup": False,
        "looptran": 0,
        "tran_converge": None,
        "no_norm": False,
        "no_input_norm": False,
    }


def _detect_format(path):
    """Return ``'toml'`` if *path* ends with ``.toml``, otherwise ``'json'``."""
    if path and path.lower().endswith(".toml"):
        return "toml"
    return "json"


def _toml_load(path):
    """Load a TOML file and return it as a dict.

    Uses the stdlib ``tomllib`` module (Python 3.11+) with a graceful
    fallback to the third-party ``tomli`` package for older Python versions.
    Raises SystemExit with a clear message on any error.
    """
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            print(
                "Error: TOML support requires Python 3.11+ or the 'tomli' package "
                "(pip install tomli)."
            )
            sys.exit(1)
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        print(f"Error: config file not found: {path}")
        sys.exit(1)
    except tomllib.TOMLDecodeError as exc:
        print(f"Error: invalid TOML in config file '{path}': {exc}")
        sys.exit(1)


def _dict_to_toml(d):
    """Serialize a flat dict to a TOML string.

    Supported value types: ``str``, ``int``, ``float``, ``bool``, ``list``
    of strings.  ``None`` values are omitted because TOML has no null type;
    omitting them is equivalent — argparse will fall back to its own
    built-in default (also ``None``) for any key not present in the file.
    """
    lines = []
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, int):
            lines.append(f"{key} = {value}")
        elif isinstance(value, float):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        elif isinstance(value, list):
            items = ", ".join(
                '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
                for v in value
            )
            lines.append(f"{key} = [{items}]")
    return "\n".join(lines) + "\n"


def _load_config_file(path):
    """Load a JSON or TOML config file and return it as a dict.

    The format is determined by the file extension: ``.toml`` → TOML,
    anything else → JSON.  Raises SystemExit with an error message if the
    file cannot be read or contains invalid content.
    """
    if _detect_format(path) == "toml":
        data = _toml_load(path)
    else:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            print(f"Error: config file not found: {path}")
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON in config file '{path}': {exc}")
            sys.exit(1)
    if not isinstance(data, dict):
        print(f"Error: config file '{path}' must contain a JSON object (dict)")
        sys.exit(1)
    return data


def _write_config_file(defaults, path):
    """Write *defaults* as a JSON or TOML config file to *path*.

    The format is determined by the file extension: ``.toml`` → TOML,
    anything else → JSON.  If *path* is ``None`` or ``"-"``, writes JSON
    to stdout instead.
    """
    if path is None or path == "-":
        print(json.dumps(defaults, indent=2))
        return
    if _detect_format(path) == "toml":
        content = _dict_to_toml(defaults)
    else:
        content = json.dumps(defaults, indent=2) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"Config written to: {path}")
