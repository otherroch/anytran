import sys
from pathlib import Path
import numpy as np

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anytran import tts


# ---------------------------------------------------------------------------
# _normalize_fish_model_name
# ---------------------------------------------------------------------------

def test_normalize_fish_model_name_default():
    """None or empty string should return the default s1-mini model."""
    assert tts._normalize_fish_model_name(None) == "fishaudio/openaudio-s1-mini"
    assert tts._normalize_fish_model_name("") == "fishaudio/openaudio-s1-mini"


def test_normalize_fish_model_name_alias():
    """fishaudio/s1-mini is an alias for fishaudio/openaudio-s1-mini."""
    assert tts._normalize_fish_model_name("fishaudio/s1-mini") == "fishaudio/openaudio-s1-mini"


def test_normalize_fish_model_name_canonical_passthrough():
    """Canonical names should be returned unchanged."""
    assert tts._normalize_fish_model_name("fishaudio/openaudio-s1-mini") == "fishaudio/openaudio-s1-mini"
    assert tts._normalize_fish_model_name("fishaudio/fish-speech-1.5") == "fishaudio/fish-speech-1.5"


# ---------------------------------------------------------------------------
# fish_tts – unavailable package
# ---------------------------------------------------------------------------

def test_fish_tts_not_available_returns_false(tmp_path, monkeypatch):
    """fish_tts returns False when fish-speech package is not installed."""
    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", False)

    result = tts.fish_tts(
        "Hello world",
        "fishaudio/openaudio-s1-mini",
        str(tmp_path / "output.wav"),
        verbose=False,
    )

    assert result is False


def test_fish_import_block_catches_non_import_errors(monkeypatch):
    """
    FISH_TTS_AVAILABLE must be False and all names must be set to None when
    fish-speech imports raise non-ImportError exceptions (e.g. FileNotFoundError
    from pyrootutils.setup_root, or AttributeError from torch._inductor accesses
    that fish-speech modules perform at import time).
    """
    # The module-level try/except has already run; we verify the module is in a
    # consistent state even when FISH_TTS_AVAILABLE is False.
    assert hasattr(tts, "FISH_TTS_AVAILABLE")
    assert hasattr(tts, "_FishTTSInferenceEngine")
    assert hasattr(tts, "_fish_load_decoder_model")
    assert hasattr(tts, "_fish_launch_llama_queue")
    assert hasattr(tts, "_FishServeReferenceAudio")
    assert hasattr(tts, "_FishServeTTSRequest")

    # Simulate a fresh import where fish-speech raises RuntimeError (like pyrootutils)
    import importlib, sys, types

    # Back up and remove cached modules so we can re-exec the import block
    _sentinel = object()

    original_available = tts.FISH_TTS_AVAILABLE

    # Build a fake fish_speech package whose sub-module raises RuntimeError
    fake_fs = types.ModuleType("fish_speech")
    fake_ie = types.ModuleType("fish_speech.inference_engine")

    # Make TTSInferenceEngine import succeed ...
    fake_ie.TTSInferenceEngine = object()

    # ... but the dac.inference module raises RuntimeError at import time
    fake_models = types.ModuleType("fish_speech.models")
    fake_dac = types.ModuleType("fish_speech.models.dac")
    fake_dac_inf = types.ModuleType("fish_speech.models.dac.inference")

    original_modules = {}
    for mod in ["fish_speech", "fish_speech.inference_engine", "fish_speech.models",
                "fish_speech.models.dac", "fish_speech.models.dac.inference"]:
        original_modules[mod] = sys.modules.pop(mod, _sentinel)

    sys.modules["fish_speech"] = fake_fs
    sys.modules["fish_speech.inference_engine"] = fake_ie
    sys.modules["fish_speech.models"] = fake_models
    sys.modules["fish_speech.models.dac"] = fake_dac

    # Trigger RuntimeError when fish_speech.models.dac.inference is imported
    class _RaisingFinder:
        def find_module(self, name, path=None):
            if name == "fish_speech.models.dac.inference":
                return self
            return None
        def load_module(self, name):
            raise RuntimeError("pyrootutils could not find .project-root")

    finder = _RaisingFinder()
    sys.meta_path.insert(0, finder)

    try:
        ns = {}
        exec("""
try:
    from fish_speech.inference_engine import TTSInferenceEngine as _FishTTSInferenceEngine
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue as _fish_launch_llama_queue
    from fish_speech.utils.schema import ServeReferenceAudio as _FishServeReferenceAudio
    from fish_speech.utils.schema import ServeTTSRequest as _FishServeTTSRequest
    import pyrootutils as _pyrootutils
    _pyrootutils_orig = _pyrootutils.setup_root
    _pyrootutils.setup_root = lambda *a, **kw: None
    try:
        from fish_speech.models.dac.inference import load_model as _fish_load_decoder_model
    finally:
        _pyrootutils.setup_root = _pyrootutils_orig
    del _pyrootutils, _pyrootutils_orig
    FISH_TTS_AVAILABLE = True
except Exception:
    _FishTTSInferenceEngine = None
    _fish_load_decoder_model = None
    _fish_launch_llama_queue = None
    _FishServeReferenceAudio = None
    _FishServeTTSRequest = None
    FISH_TTS_AVAILABLE = False
""", ns)
        # All names must exist and be None / False
        assert ns["FISH_TTS_AVAILABLE"] is False
        assert ns["_FishTTSInferenceEngine"] is None
        assert ns["_fish_load_decoder_model"] is None
        assert ns["_fish_launch_llama_queue"] is None
        assert ns["_FishServeReferenceAudio"] is None
        assert ns["_FishServeTTSRequest"] is None
    finally:
        sys.meta_path.remove(finder)
        # Restore sys.modules
        for mod, orig in original_modules.items():
            if orig is _sentinel:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = orig


def test_fish_import_block_pyrootutils_patch_makes_dac_inference_succeed():
    """
    The import block patches pyrootutils.setup_root to a no-op before importing
    fish_speech.models.dac.inference so that the FileNotFoundError raised by a
    pip-installed fish-speech package (no .project-root marker file) does not
    prevent the import from succeeding.  FISH_TTS_AVAILABLE must be True and
    _fish_load_decoder_model must be the real load_model function.
    """
    import importlib.util
    import sys
    import types

    _sentinel = object()

    # Fake pyrootutils whose setup_root raises FileNotFoundError by default,
    # simulating a pip-installed fish-speech package without .project-root.
    fake_pyrootutils = types.ModuleType("pyrootutils")
    def _raising_setup_root(*a, **kw):
        raise FileNotFoundError(
            "Project root directory not found. Indicators: ['.project-root']"
        )
    fake_pyrootutils.setup_root = _raising_setup_root

    # A finder that creates fish_speech.models.dac.inference on the fly.
    # It mimics what the real module does: call pyrootutils.setup_root() at
    # import time.  If the import block has patched it to a no-op the load
    # succeeds; otherwise FileNotFoundError propagates.
    class _DacInferenceFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname == "fish_speech.models.dac.inference":
                return importlib.util.spec_from_loader(fullname, self)
            return None

        def create_module(self, spec):
            return None  # use default module creation

        def exec_module(self, module):
            import pyrootutils  # resolved via sys.modules → fake_pyrootutils
            pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
            module.load_model = lambda *a, **kw: None

    finder = _DacInferenceFinder()

    # Build the rest of the fake fish_speech package (all successful).
    # Intermediate packages need __path__ so Python treats them as packages.
    fake_fs = types.ModuleType("fish_speech")
    fake_fs.__path__ = []
    fake_ie = types.ModuleType("fish_speech.inference_engine")
    fake_ie.TTSInferenceEngine = type("TTSInferenceEngine", (), {})
    fake_models = types.ModuleType("fish_speech.models")
    fake_models.__path__ = []
    fake_dac = types.ModuleType("fish_speech.models.dac")
    fake_dac.__path__ = []
    fake_t2s = types.ModuleType("fish_speech.models.text2semantic")
    fake_t2s.__path__ = []
    fake_t2s_inf = types.ModuleType("fish_speech.models.text2semantic.inference")
    fake_t2s_inf.launch_thread_safe_queue = lambda *a, **kw: None
    fake_utils = types.ModuleType("fish_speech.utils")
    fake_utils.__path__ = []
    fake_schema = types.ModuleType("fish_speech.utils.schema")
    fake_schema.ServeReferenceAudio = type("ServeReferenceAudio", (), {})
    fake_schema.ServeTTSRequest = type("ServeTTSRequest", (), {})

    mods_to_swap = [
        "pyrootutils",
        "fish_speech", "fish_speech.inference_engine",
        "fish_speech.models", "fish_speech.models.dac",
        "fish_speech.models.dac.inference",
        "fish_speech.models.text2semantic",
        "fish_speech.models.text2semantic.inference",
        "fish_speech.utils", "fish_speech.utils.schema",
    ]
    original_modules = {m: sys.modules.pop(m, _sentinel) for m in mods_to_swap}

    sys.modules.update({
        "pyrootutils": fake_pyrootutils,
        "fish_speech": fake_fs,
        "fish_speech.inference_engine": fake_ie,
        "fish_speech.models": fake_models,
        "fish_speech.models.dac": fake_dac,
        "fish_speech.models.text2semantic": fake_t2s,
        "fish_speech.models.text2semantic.inference": fake_t2s_inf,
        "fish_speech.utils": fake_utils,
        "fish_speech.utils.schema": fake_schema,
    })
    sys.meta_path.insert(0, finder)

    try:
        ns = {}
        exec("""
try:
    from fish_speech.inference_engine import TTSInferenceEngine as _FishTTSInferenceEngine
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue as _fish_launch_llama_queue
    from fish_speech.utils.schema import ServeReferenceAudio as _FishServeReferenceAudio
    from fish_speech.utils.schema import ServeTTSRequest as _FishServeTTSRequest
    import pyrootutils as _pyrootutils
    _pyrootutils_orig = _pyrootutils.setup_root
    _pyrootutils.setup_root = lambda *a, **kw: None
    try:
        from fish_speech.models.dac.inference import load_model as _fish_load_decoder_model
    finally:
        _pyrootutils.setup_root = _pyrootutils_orig
    del _pyrootutils, _pyrootutils_orig
    FISH_TTS_AVAILABLE = True
except Exception:
    _FishTTSInferenceEngine = None
    _fish_load_decoder_model = None
    _fish_launch_llama_queue = None
    _FishServeReferenceAudio = None
    _FishServeTTSRequest = None
    FISH_TTS_AVAILABLE = False
""", ns)
        # The patch must have allowed dac.inference to load successfully.
        assert ns["FISH_TTS_AVAILABLE"] is True, (
            "FISH_TTS_AVAILABLE should be True when pyrootutils.setup_root is patched"
        )
        assert ns["_fish_load_decoder_model"] is not None
        assert ns["_FishTTSInferenceEngine"] is not None
        assert ns["_fish_launch_llama_queue"] is not None
        assert ns["_FishServeReferenceAudio"] is not None
        assert ns["_FishServeTTSRequest"] is not None
        # setup_root must have been restored to the raising version after import
        assert fake_pyrootutils.setup_root is _raising_setup_root
    finally:
        sys.meta_path.remove(finder)
        for mod, orig in original_modules.items():
            if orig is _sentinel:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = orig



# ---------------------------------------------------------------------------
# _load_fish_engine – torchaudio availability check
# ---------------------------------------------------------------------------

def test_load_fish_engine_missing_torchaudio_returns_none(tmp_path, monkeypatch, capsys):
    """
    _load_fish_engine returns None and prints a helpful install hint when
    torchaudio is not importable.  The torchaudio check runs before any
    network I/O (huggingface_hub download), so no mocking of HF hub is needed.
    """
    import sys
    import types

    _sentinel = object()
    original_torchaudio = sys.modules.pop("torchaudio", _sentinel)

    # Finder that makes 'import torchaudio' raise ImportError
    class _BrokenTorchaudioFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname == "torchaudio":
                raise ImportError("No module named 'torchaudio'")
            return None

    finder = _BrokenTorchaudioFinder()
    sys.meta_path.insert(0, finder)

    # Provide a minimal torch stub so TORCH_AVAILABLE check passes
    fake_torch = types.ModuleType("torch")
    _orig_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch

    try:
        monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
        monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(tts, "torch", fake_torch)
        monkeypatch.setattr(tts, "_fish_model_cache", {})

        result = tts._load_fish_engine("fishaudio/openaudio-s1-mini", verbose=False)
        assert result is None

        captured = capsys.readouterr()
        assert "torchaudio" in captured.out
        assert "pip install" in captured.out
    finally:
        sys.meta_path.remove(finder)
        if original_torchaudio is _sentinel:
            sys.modules.pop("torchaudio", None)
        else:
            sys.modules["torchaudio"] = original_torchaudio
        if _orig_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = _orig_torch


def test_load_fish_engine_missing_list_audio_backends(tmp_path, monkeypatch, capsys):
    """
    When torchaudio does NOT have list_audio_backends() — whether because it's an
    old release (< 0.12) or a new nightly that removed the API — _load_fish_engine
    patches a shim onto the module so that the fish-speech ReferenceLoader can still
    construct itself.  The engine must be returned successfully (not None) when all
    other dependencies are satisfied.
    """
    import sys
    import types

    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    # Stub a torchaudio that is importable but does NOT have list_audio_backends
    fake_old_torchaudio = types.ModuleType("torchaudio")
    # deliberately omit list_audio_backends

    _orig_ta = sys.modules.get("torchaudio")
    sys.modules["torchaudio"] = fake_old_torchaudio

    # Minimal torch stub
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_mps = types.SimpleNamespace(is_available=lambda: False)
    fake_backends = types.SimpleNamespace(mps=fake_mps)
    fake_torch.cuda = fake_cuda
    fake_torch.backends = fake_backends
    fake_torch.bfloat16 = "bfloat16"
    _orig_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch

    # Stub out huggingface_hub snapshot_download
    fake_checkpoint = tmp_path / "model"
    fake_checkpoint.mkdir()
    (fake_checkpoint / "codec.pth").touch()

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *a, **kw: str(fake_checkpoint)
    _orig_hf = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake_hf

    mock_engine = _make_mock_engine()

    try:
        monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
        monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(tts, "torch", fake_torch)
        monkeypatch.setattr(tts, "_fish_model_cache", {})
        monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
        monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
        monkeypatch.setattr(tts, "_fish_launch_llama_queue", lambda **kw: object())
        monkeypatch.setattr(tts, "_fish_load_decoder_model", lambda **kw: object())
        monkeypatch.setattr(tts, "_FishTTSInferenceEngine", lambda **kw: mock_engine)

        result = tts._load_fish_engine("fishaudio/openaudio-s1-mini", verbose=False)

        # The shim must have been patched onto the module
        assert hasattr(fake_old_torchaudio, "list_audio_backends"), (
            "list_audio_backends shim was not applied to torchaudio"
        )
        # And the engine must be returned, not None
        assert result is mock_engine, "Expected engine to be returned, got None"

        # No error message should appear
        captured = capsys.readouterr()
        assert "too old" not in captured.out
        assert "upgrade" not in captured.out.lower()
    finally:
        if _orig_ta is None:
            sys.modules.pop("torchaudio", None)
        else:
            sys.modules["torchaudio"] = _orig_ta
        if _orig_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = _orig_torch
        if _orig_hf is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = _orig_hf


def test_load_fish_engine_prints_full_traceback_on_failure(tmp_path, monkeypatch, capsys):
    """
    When _fish_load_decoder_model raises an unexpected exception,
    _load_fish_engine prints both the short summary line (stdout) and the full
    traceback (stderr) so users can diagnose environment problems such as a
    PyTorch version that is too old for the mmap=True parameter.
    """
    import sys
    import types

    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)

    # Minimal torch stub with cuda / backends stubs
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_mps  = types.SimpleNamespace(is_available=lambda: False)
    fake_backends = types.SimpleNamespace(mps=fake_mps)
    fake_torch.cuda = fake_cuda
    fake_torch.backends = fake_backends
    fake_torch.bfloat16 = "bfloat16"
    monkeypatch.setattr(tts, "torch", fake_torch)

    # torchaudio must be importable (so we get past that check)
    fake_torchaudio = types.ModuleType("torchaudio")
    fake_torchaudio.list_audio_backends = lambda: ["soundfile"]  # satisfy version check
    _orig_ta = sys.modules.get("torchaudio")
    sys.modules["torchaudio"] = fake_torchaudio

    # Stub out huggingface_hub snapshot_download
    fake_checkpoint = tmp_path / "model"
    fake_checkpoint.mkdir()
    (fake_checkpoint / "codec.pth").touch()

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *a, **kw: str(fake_checkpoint)
    _orig_hf = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake_hf

    # LLaMA queue loads fine; decoder raises a descriptive error
    monkeypatch.setattr(tts, "_fish_launch_llama_queue", lambda **kw: object())

    def _failing_load_model(**kw):
        raise RuntimeError("mmap parameter requires PyTorch >= 2.1")

    monkeypatch.setattr(tts, "_fish_load_decoder_model", _failing_load_model)

    try:
        result = tts._load_fish_engine("fishaudio/openaudio-s1-mini", verbose=False)
    finally:
        if _orig_hf is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = _orig_hf
        if _orig_ta is None:
            sys.modules.pop("torchaudio", None)
        else:
            sys.modules["torchaudio"] = _orig_ta

    assert result is None

    captured = capsys.readouterr()
    # Short summary goes to stdout
    assert "Failed to load engine" in captured.out
    # Full traceback goes to stderr
    assert "Traceback" in captured.err
    assert "RuntimeError" in captured.err
    assert "mmap parameter requires PyTorch" in captured.err




def _make_mock_engine(sample_rate=44100, audio_length=44100):
    """Return a mock TTSInferenceEngine that yields a 1-second silence clip."""

    class MockInferenceResult:
        def __init__(self, code, audio, error=None):
            self.code = code
            self.audio = audio
            self.error = error

    class MockEngine:
        def __init__(self, sr=sample_rate, length=audio_length):
            self._sr = sr
            self._length = length

        def inference(self, request):
            audio = np.zeros(self._length, dtype=np.float32)
            yield MockInferenceResult(code="final", audio=(self._sr, audio))

    return MockEngine()


def _make_mock_fish_schema():
    """Return lightweight stand-ins for ServeReferenceAudio and ServeTTSRequest."""

    class FakeReferenceAudio:
        def __init__(self, audio, text):
            self.audio = audio
            self.text = text

    class FakeTTSRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    return FakeReferenceAudio, FakeTTSRequest


# ---------------------------------------------------------------------------
# fish_tts – successful synthesis (no voice cloning)
# ---------------------------------------------------------------------------

def test_fish_tts_basic_synthesis(tmp_path, monkeypatch):
    """fish_tts saves a WAV file and returns True on success."""
    output_wav = tmp_path / "output.wav"
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)

    engine = _make_mock_engine()
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: engine)

    result = tts.fish_tts(
        "Hello world",
        "fishaudio/openaudio-s1-mini",
        str(output_wav),
        verbose=False,
    )

    assert result is True
    assert output_wav.exists()


def test_fish_tts_uses_model_cache(tmp_path, monkeypatch):
    """fish_tts caches the engine so _load_fish_engine is only called once."""
    load_calls = {"count": 0}
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        load_calls["count"] += 1
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    model = "fishaudio/openaudio-s1-mini"
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"

    tts.fish_tts("Hello", model, str(out1), verbose=False)
    tts.fish_tts("World", model, str(out2), verbose=False)

    assert load_calls["count"] == 1  # loaded only once


def test_fish_tts_alias_resolved_before_cache_lookup(tmp_path, monkeypatch):
    """The s1-mini alias is resolved before the cache is consulted."""
    loaded_names = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        loaded_names.append(model_name)
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    out = tmp_path / "out.wav"
    tts.fish_tts("Hello", "fishaudio/s1-mini", str(out), verbose=False)

    assert loaded_names == ["fishaudio/openaudio-s1-mini"]


# ---------------------------------------------------------------------------
# fish_tts – voice cloning (with reference audio)
# ---------------------------------------------------------------------------

def test_fish_tts_voice_cloning(tmp_path, monkeypatch):
    """fish_tts passes reference audio to the engine when provided."""
    ref_received = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    class MockEngineCapturingRef:
        def inference(self, request):
            ref_received.extend(request.references)
            audio = np.zeros(44100, dtype=np.float32)

            class R:
                code = "final"
                audio = (44100, np.zeros(44100, dtype=np.float32))
                error = None

            yield R()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: MockEngineCapturingRef())

    ref_audio = np.clip(np.random.randn(16000), -1.0, 1.0).astype(np.float32)  # float32 in [-1.0, 1.0]
    out = tmp_path / "out.wav"

    result = tts.fish_tts(
        "Clone this voice",
        "fishaudio/openaudio-s1-mini",
        str(out),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        reference_text="This is the reference",
        verbose=False,
    )

    assert result is True
    assert len(ref_received) == 1
    assert ref_received[0].text == "This is the reference"


def test_fish_tts_int16_reference_audio(tmp_path, monkeypatch):
    """fish_tts normalises int16 reference audio to float32 before encoding."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())

    # int16 PCM reference audio (as used by the rest of anytran)
    ref_audio = (np.clip(np.random.randn(16000), -1.0, 1.0) * 32767).astype(np.int16)
    out = tmp_path / "out.wav"

    result = tts.fish_tts(
        "Hello",
        "fishaudio/openaudio-s1-mini",
        str(out),
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        verbose=False,
    )

    assert result is True


# ---------------------------------------------------------------------------
# fish_tts – fish-speech-1.5 model
# ---------------------------------------------------------------------------

def test_fish_tts_fish_speech_15(tmp_path, monkeypatch):
    """fish_tts works with the fishaudio/fish-speech-1.5 model."""
    loaded_names = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        loaded_names.append(model_name)
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    out = tmp_path / "out.wav"
    result = tts.fish_tts("Hello", "fishaudio/fish-speech-1.5", str(out), verbose=False)

    assert result is True
    assert loaded_names == ["fishaudio/fish-speech-1.5"]


# ---------------------------------------------------------------------------
# fish_tts – engine load failure
# ---------------------------------------------------------------------------

def test_fish_tts_engine_load_failure_returns_false(tmp_path, monkeypatch):
    """fish_tts returns False when the engine cannot be loaded."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: None)

    result = tts.fish_tts(
        "Hello",
        "fishaudio/openaudio-s1-mini",
        str(tmp_path / "output.wav"),
        verbose=False,
    )

    assert result is False


# ---------------------------------------------------------------------------
# synthesize_tts_pcm – fish backend
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_fish_backend(monkeypatch):
    """synthesize_tts_pcm returns an int16 numpy array when using fish backend."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())

    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


def test_synthesize_tts_pcm_fish_default_model(monkeypatch):
    """synthesize_tts_pcm uses the default fish model when given a piper-style name."""
    loaded_names = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    def fake_load(model_name, verbose=False):
        loaded_names.append(model_name)
        return _make_mock_engine()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", fake_load)

    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="fish",
        voice_model="en_US-lessac-medium",  # piper-style name – should be replaced
        verbose=False,
    )

    assert result is not None
    assert loaded_names[0] == "fishaudio/openaudio-s1-mini"


# ---------------------------------------------------------------------------
# synthesize_tts_pcm_with_cloning – fish backend with voice_match
# ---------------------------------------------------------------------------

def test_synthesize_tts_pcm_with_cloning_fish_voice_match(monkeypatch):
    """synthesize_tts_pcm_with_cloning does zero-shot cloning for fish + voice_match."""
    ref_texts_received = []
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    class MockEngineCapturingRef:
        def inference(self, request):
            for ref in request.references:
                ref_texts_received.append(ref.text)
            audio = np.zeros(44100, dtype=np.float32)

            class R:
                code = "final"
                audio = (44100, np.zeros(44100, dtype=np.float32))
                error = None

            yield R()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine",
                        lambda model_name, verbose=False: MockEngineCapturingRef())

    ref_audio = (np.clip(np.random.randn(16000), -1.0, 1.0) * 32767).astype(np.int16)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Synthesize this",
        rate=16000,
        output_lang="en",
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        reference_text="Reference text for cloning",
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        voice_match=True,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    assert len(ref_texts_received) == 1
    assert ref_texts_received[0] == "Reference text for cloning"


def test_synthesize_tts_pcm_with_cloning_fish_no_voice_match(monkeypatch):
    """With fish backend but voice_match=False, basic synthesis is used (no cloning)."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())
    # Reset piper voice cache so we do not accidentally pick up a cached piper voice
    monkeypatch.setattr(tts, "_cached_matched_voice", None)
    monkeypatch.setattr(tts, "_cached_output_lang", None)

    ref_audio = (np.clip(np.random.randn(16000), -1.0, 1.0) * 32767).astype(np.int16)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        voice_match=False,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


def test_synthesize_tts_pcm_with_cloning_fish_voice_match_no_ref(monkeypatch):
    """With voice_match but no reference audio, basic synthesis is used."""
    FakeRefAudio, FakeTTSRequest = _make_mock_fish_schema()

    monkeypatch.setattr(tts, "FISH_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "_fish_model_cache", {})
    monkeypatch.setattr(tts, "_FishServeReferenceAudio", FakeRefAudio)
    monkeypatch.setattr(tts, "_FishServeTTSRequest", FakeTTSRequest)
    monkeypatch.setattr(tts, "_load_fish_engine", lambda model_name, verbose=False: _make_mock_engine())
    monkeypatch.setattr(tts, "_cached_matched_voice", None)
    monkeypatch.setattr(tts, "_cached_output_lang", None)

    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=None,  # no reference audio
        voice_backend="fish",
        voice_model="fishaudio/openaudio-s1-mini",
        voice_match=True,
        verbose=False,
    )

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
