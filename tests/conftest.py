"""
conftest.py — Pre-import real modules so that test_looptran.py's and
test_pipeline_stages.py's module-level stubs don't replace them before our
coverage tests can use them.

Both files install stubs for several anytran modules at *collection* time
(i.e., at the module level, not inside fixtures).  The stubs only replace a
module when it is NOT already in sys.modules (``if _mod_name not in
sys.modules``).  By importing the real implementations here — in the conftest
that pytest loads *before* collecting any test file — we ensure the real
modules are already registered so the conditional stubs are skipped.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def ensure_torch_stub_attributes():
    """
    After test collection several modules stub ``torch`` as a bare
    ``types.ModuleType``.  Runtime calls to librosa can trigger
    ``torch.Tensor`` lookups even when torch is not installed, because the
    stub *exists* in sys.modules and librosa therefore thinks torch is
    available.  Add the minimum attributes so those lookups don't raise.
    """
    import sys
    torch_stub = sys.modules.get("torch")
    if torch_stub is not None and not hasattr(torch_stub, "Tensor"):
        torch_stub.Tensor = type("MockTensor", (), {})
    yield


# stdlib / third-party modules that are stubbable but actually installed
import librosa  # noqa: F401 — must be before test_looptran stubs torch (librosa checks torch at import time)
import paho.mqtt.client  # noqa: F401
# fastapi and uvicorn are imported INSIDE run_web_server() (not at module level
# in web_server.py), so we pre-import them here before the stubs take effect.
import fastapi  # noqa: F401
import uvicorn  # noqa: F401

# anytran modules that the new coverage tests exercise directly
import anytran.whisper_backend  # noqa: F401
import anytran.vad  # noqa: F401
import anytran.mqtt_client  # noqa: F401
import anytran.voice_matcher  # noqa: F401
import anytran.tts  # noqa: F401
import anytran.chatlog  # noqa: F401
import anytran.certs  # noqa: F401

# Pre-import web_server (and its transitive deps: processing, mqtt_client)
# so that test_looptran.py's stubs can't replace them.
import anytran.web_server  # noqa: F401

# -------------------------------------------------------------------------
# Save real config function references BEFORE test_looptran.py replaces
# them with MagicMock instances (it does setattr on the live module).
# -------------------------------------------------------------------------
import anytran.config as _anytran_config_module

_real_config_funcs = {
    name: getattr(_anytran_config_module, name)
    for name in (
        "set_whisper_backend",
        "get_whisper_backend",
        "set_whisper_cpp_config",
        "get_whisper_cpp_config",
        "set_whispercpp_cli_detect_lang",
        "get_whispercpp_cli_detect_lang",
        "set_whispercpp_force_cli",
        "get_whispercpp_force_cli",
        "set_whisper_ctranslate2_config",
        "get_whisper_ctranslate2_config",
        "_parse_whisper_ctranslate2_device_index",
    )
}

# Save real utils functions similarly (test_looptran stubs resolve_path_with_fallback)
import anytran.utils as _anytran_utils_module
import anytran.whisper_backend as _anytran_wb_module
import anytran.certs as _anytran_certs_module
import anytran.web_server as _anytran_web_server_module

_real_certs_funcs = {
    "generate_self_signed_cert": _anytran_certs_module.generate_self_signed_cert,
}

# Save real text_translator functions before test_looptran.py replaces them
import anytran.text_translator as _anytran_tt_module
_real_text_translator_funcs = {
    name: getattr(_anytran_tt_module, name)
    for name in (
        "translate_text",
        "set_translation_backend",
        "get_translation_backend",
        "translate_text_googletrans",
        "translate_text_libretranslate",
    )
}

_real_web_server_funcs = {
    "run_web_server": _anytran_web_server_module.run_web_server,
    "_serialize_tts_segments": _anytran_web_server_module._serialize_tts_segments,
}

_real_whisper_backend_funcs = {
    name: getattr(_anytran_wb_module, name)
    for name in (
        "_derive_whispercpp_model_name",
        "_get_with_override",
        "_extract_detected_language_from_output",
        "_extract_detected_language_from_result",
        "get_effective_backend",
        "is_hallucination",
        "download_whisper_cpp_model",
        "_call_with_native_output_capture",
        "_resolve_faster_whisper_model_path",
        "_resolve_ctranslate2_model_path",
        "_detect_language_whispercpp_cli_from_wav",
    )
}
