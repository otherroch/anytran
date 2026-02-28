import os

DEFAULT_WHISPERCPP_BIN = os.environ.get(
    "WHISPER_CPP_BIN", os.path.join("whispercpp", "whisper-cli.exe")
)
DEFAULT_WHISPERCPP_MODEL = os.environ.get(
    "WHISPER_CPP_MODEL", os.path.join("models", "ggml-medium.bin")
)
DEFAULT_WHISPERCPP_THREADS = int(os.environ.get("WHISPER_CPP_THREADS", "1"))
DEFAULT_WHISPER_CTRANSLATE2_MODEL = os.environ.get(
    "WHISPER_CTRANSLATE2_MODEL", "small"
)
DEFAULT_WHISPER_CTRANSLATE2_DEVICE = os.environ.get(
    "WHISPER_CTRANSLATE2_DEVICE", "auto"
)
_whisper_ctranslate2_device_index_env = os.environ.get("WHISPER_CTRANSLATE2_DEVICE_INDEX")

def _parse_whisper_ctranslate2_device_index(env_value):
    if env_value is None:
        return None
    try:
        return int(env_value)
    except ValueError:
        print(
            f"Invalid value '{env_value}' for WHISPER_CTRANSLATE2_DEVICE_INDEX "
            "environment variable. Expected an integer. Using default: None"
        )
        return None

DEFAULT_WHISPER_CTRANSLATE2_DEVICE_INDEX = _parse_whisper_ctranslate2_device_index(
    _whisper_ctranslate2_device_index_env
)
DEFAULT_WHISPER_CTRANSLATE2_COMPUTE_TYPE = os.environ.get(
    "WHISPER_CTRANSLATE2_COMPUTE_TYPE", "default"
)

_whisper_cpp_config = {
    "bin_path": DEFAULT_WHISPERCPP_BIN,
    "model_path": DEFAULT_WHISPERCPP_MODEL,
    "threads": DEFAULT_WHISPERCPP_THREADS,
}

_whisper_ctranslate2_config = {
    "model_name": DEFAULT_WHISPER_CTRANSLATE2_MODEL,
    "device": DEFAULT_WHISPER_CTRANSLATE2_DEVICE,
    "device_index": DEFAULT_WHISPER_CTRANSLATE2_DEVICE_INDEX,
    "compute_type": DEFAULT_WHISPER_CTRANSLATE2_COMPUTE_TYPE,
}

_whisper_backend = os.environ.get("WHISPER_BACKEND", "whispercpp")
_whispercpp_cli_detect_lang = False
_whispercpp_force_cli = False


def set_whisper_backend(backend):
    global _whisper_backend
    _whisper_backend = backend


def get_whisper_backend():
    return _whisper_backend


def set_whispercpp_cli_detect_lang(enabled):
    global _whispercpp_cli_detect_lang
    _whispercpp_cli_detect_lang = bool(enabled)


def get_whispercpp_cli_detect_lang():
    return _whispercpp_cli_detect_lang


def set_whispercpp_force_cli(enabled):
    global _whispercpp_force_cli
    _whispercpp_force_cli = bool(enabled)


def get_whispercpp_force_cli():
    return _whispercpp_force_cli


def set_whisper_cpp_config(bin_path=None, model_path=None, threads=None):
    if bin_path is not None:
        _whisper_cpp_config["bin_path"] = bin_path
    if model_path is not None:
        _whisper_cpp_config["model_path"] = model_path
    if threads is not None:
        _whisper_cpp_config["threads"] = threads


def get_whisper_cpp_config():
    return dict(_whisper_cpp_config)


def set_whisper_ctranslate2_config(model_name=None, device=None, device_index=None, compute_type=None):
    """Update the default configuration for the CTranslate2 Whisper backend.

    Each argument is optional; only parameters that are not ``None`` will
    overwrite the corresponding key in the internal ``_whisper_ctranslate2_config``
    dictionary.

    :param model_name: Name of the CTranslate2 Whisper model to use
        (for example, ``"small"``). If ``None``, the existing value is kept.
    :param device: Target device type (for example, ``"cpu"``, ``"cuda"``,
        or ``"auto"``). If ``None``, the existing value is kept.
    :param device_index: Index of the device to use (such as a GPU index) or
        ``None`` to leave the current index unchanged.
    :param compute_type: CTranslate2 compute type string (for example,
        ``"default"``, ``"int8"``). If ``None``, the existing value is kept.
    """
    for key, value in (
        ("model_name", model_name),
        ("device", device),
        ("device_index", device_index),
        ("compute_type", compute_type),
    ):
        if value is not None:
            _whisper_ctranslate2_config[key] = value


def get_whisper_ctranslate2_config():
    return dict(_whisper_ctranslate2_config)
