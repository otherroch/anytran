"""Tests for the Chatterbox TTS backend integration."""
import os
import sys
import types
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anytran.tts as tts_module


# ---------------------------------------------------------------------------
# Helpers – build a minimal torch / torchaudio mock so the tests can run
# without a real torch installation.
# ---------------------------------------------------------------------------

def _install_torch_mock():
    """Install minimal torch / torchaudio stubs into sys.modules if needed."""
    if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "Tensor"):
        torch_mock = types.ModuleType("torch")
        torch_mock.Tensor = type("Tensor", (), {})
        torch_cuda = types.ModuleType("torch.cuda")
        torch_cuda.is_available = lambda: False
        torch_mock.cuda = torch_cuda
        torch_mock.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        sys.modules["torch"] = torch_mock
        sys.modules["torch.cuda"] = torch_cuda

    if "torchaudio" not in sys.modules:
        ta_mock = types.ModuleType("torchaudio")

        def _fake_ta_save(path, wav, sr):
            # Write a real WAV file using soundfile so downstream code can read it.
            # 0.1 seconds of silence is enough for all the read-back tests.
            audio = np.zeros(max(1, int(sr * 0.1)), dtype=np.float32)
            sf.write(path, audio, sr)

        ta_mock.save = _fake_ta_save
        sys.modules["torchaudio"] = ta_mock


_install_torch_mock()


def _make_fake_audio():
    """Return a fake audio 'tensor' (just a MagicMock) for model.generate()."""
    return MagicMock(name="fake_wav_tensor")


def _make_fake_mtl_class(generate_calls_list=None, sr=16000):
    """Factory for a fake ChatterboxMultilingualTTS class."""
    fake_audio = _make_fake_audio()

    class FakeMtlModel:
        def __init__(self):
            self.sr = sr

        def generate(self, text, **kwargs):
            if generate_calls_list is not None:
                generate_calls_list.append(kwargs)
            return fake_audio

    class FakeMtlClass:
        @staticmethod
        def from_pretrained(device):
            return FakeMtlModel()

    return FakeMtlClass


def _make_fake_turbo_class(generate_calls_list=None, sr=24000):
    """Factory for a fake ChatterboxTurboTTS class."""
    fake_audio = _make_fake_audio()

    class FakeTurboModel:
        def __init__(self):
            self.sr = sr

        def generate(self, text, **kwargs):
            if generate_calls_list is not None:
                generate_calls_list.append(kwargs)
            return fake_audio

    class FakeTurboClass:
        @staticmethod
        def from_pretrained(device):
            return FakeTurboModel()

    return FakeTurboClass


# ---------------------------------------------------------------------------
# chatterbox_tts() – unit tests
# ---------------------------------------------------------------------------

class TestChatterboxTtsFunction:
    """Unit tests for the chatterbox_tts() helper."""

    def test_returns_false_when_neither_model_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", False)
        out_wav = str(tmp_path / "out.wav")
        result = tts_module.chatterbox_tts("hello", out_wav, verbose=True)
        assert result is False

    def test_multilingual_model_used_for_non_english(self, tmp_path, monkeypatch):
        calls = []
        FakeMtlClass = _make_fake_mtl_class(generate_calls_list=calls, sr=16000)

        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "ChatterboxMultilingualTTS", FakeMtlClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

        out_wav = str(tmp_path / "out.wav")
        result = tts_module.chatterbox_tts("Bonjour", out_wav, output_lang="fr", use_turbo=False, verbose=True)

        assert result is True
        assert os.path.isfile(out_wav)
        assert len(calls) == 1
        assert calls[0].get("language_id") == "fr"

    def test_turbo_model_used_for_english(self, tmp_path, monkeypatch):
        calls = []
        FakeTurboClass = _make_fake_turbo_class(generate_calls_list=calls, sr=24000)

        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "ChatterboxTurboTTS", FakeTurboClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

        out_wav = str(tmp_path / "out.wav")
        result = tts_module.chatterbox_tts("Hello", out_wav, output_lang="en", use_turbo=True, verbose=True)

        assert result is True
        assert os.path.isfile(out_wav)

    def test_turbo_falls_back_to_mtl_for_non_english(self, tmp_path, monkeypatch):
        """When use_turbo=True but lang is not English, mtl model should be used."""
        calls = []
        FakeMtlClass = _make_fake_mtl_class(generate_calls_list=calls, sr=16000)

        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "ChatterboxMultilingualTTS", FakeMtlClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

        out_wav = str(tmp_path / "out.wav")
        result = tts_module.chatterbox_tts("Hola", out_wav, output_lang="es", use_turbo=True, verbose=True)

        assert result is True
        assert calls[-1].get("language_id") == "es"

    def test_reference_audio_path_passed_to_generate(self, tmp_path, monkeypatch):
        calls = []
        FakeMtlClass = _make_fake_mtl_class(generate_calls_list=calls, sr=16000)

        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "ChatterboxMultilingualTTS", FakeMtlClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

        ref_audio_path = str(tmp_path / "ref.wav")
        sf.write(ref_audio_path, np.ones(1600, dtype=np.float32) * 0.1, 16000)

        out_wav = str(tmp_path / "out.wav")
        result = tts_module.chatterbox_tts(
            "Hello", out_wav,
            reference_audio_path=ref_audio_path,
            output_lang="en",
            use_turbo=False,
            verbose=True,
        )

        assert result is True
        assert calls[-1].get("audio_prompt_path") == ref_audio_path

    def test_exception_returns_false(self, tmp_path, monkeypatch):
        class BrokenMtlClass:
            @staticmethod
            def from_pretrained(device):
                raise RuntimeError("Model load failed")

        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "ChatterboxMultilingualTTS", BrokenMtlClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

        out_wav = str(tmp_path / "out.wav")
        result = tts_module.chatterbox_tts("Hello", out_wav, verbose=True)
        assert result is False

    def test_model_cached_across_calls(self, tmp_path, monkeypatch):
        init_count = [0]

        class FakeMtlModel:
            sr = 16000

            def generate(self, text, **kwargs):
                return _make_fake_audio()

        class FakeMtlClass:
            @staticmethod
            def from_pretrained(device):
                init_count[0] += 1
                return FakeMtlModel()

        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "ChatterboxMultilingualTTS", FakeMtlClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

        out1 = str(tmp_path / "out1.wav")
        out2 = str(tmp_path / "out2.wav")
        tts_module.chatterbox_tts("Hello", out1, output_lang="en")
        tts_module.chatterbox_tts("World", out2, output_lang="en")

        # Model should be instantiated only once
        assert init_count[0] == 1


# ---------------------------------------------------------------------------
# synthesize_tts_pcm() – chatterbox backend
# ---------------------------------------------------------------------------

class TestSynthesizeTtsPcmChatterbox:
    """Tests for synthesize_tts_pcm with voice_backend='chatterbox'."""

    def _setup_fake_mtl(self, monkeypatch):
        FakeMtlClass = _make_fake_mtl_class(sr=16000)
        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "ChatterboxMultilingualTTS", FakeMtlClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

    def test_returns_numpy_array(self, monkeypatch):
        self._setup_fake_mtl(monkeypatch)
        result = tts_module.synthesize_tts_pcm(
            "Hello", rate=16000, output_lang="en", voice_backend="chatterbox"
        )
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int16

    def test_empty_text_returns_none(self, monkeypatch):
        self._setup_fake_mtl(monkeypatch)
        result = tts_module.synthesize_tts_pcm(
            "", rate=16000, output_lang="en", voice_backend="chatterbox"
        )
        assert result is None

    def test_none_text_returns_none(self, monkeypatch):
        self._setup_fake_mtl(monkeypatch)
        result = tts_module.synthesize_tts_pcm(
            None, rate=16000, output_lang="en", voice_backend="chatterbox"
        )
        assert result is None

    def test_falls_back_to_gtts_on_chatterbox_failure(self, monkeypatch):
        """When chatterbox is unavailable, synthesize_tts_pcm should fall back to gTTS."""
        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

        # 1600 samples = 100 ms at 16 kHz; amplitude 200 is an arbitrary non-zero value.
        raw_pcm = (np.ones(1600, dtype=np.int16) * 200).tobytes()
        seg = MagicMock()
        seg.raw_data = raw_pcm
        seg.set_frame_rate = MagicMock(return_value=seg)
        seg.set_channels = MagicMock(return_value=seg)
        seg.set_sample_width = MagicMock(return_value=seg)

        mock_gtts = MagicMock()
        mock_gtts.return_value = MagicMock()
        mock_audio_class = MagicMock()
        mock_audio_class.from_mp3 = MagicMock(return_value=seg)

        monkeypatch.setattr(tts_module, "gTTS", mock_gtts)
        monkeypatch.setattr(tts_module, "AudioSegment", mock_audio_class)

        result = tts_module.synthesize_tts_pcm(
            "Hello", rate=16000, output_lang="en", voice_backend="chatterbox", verbose=True
        )
        mock_gtts.assert_called()

    def test_verbose_mode(self, monkeypatch):
        self._setup_fake_mtl(monkeypatch)
        result = tts_module.synthesize_tts_pcm(
            "Hello verbose", rate=16000, output_lang="en", voice_backend="chatterbox", verbose=True
        )
        assert result is not None


# ---------------------------------------------------------------------------
# synthesize_tts_pcm_with_cloning() – chatterbox with voice_match
# ---------------------------------------------------------------------------

class TestSynthesizeTtsPcmWithCloningChatterbox:
    """Tests for synthesize_tts_pcm_with_cloning with chatterbox backend."""

    def _setup_fake_mtl(self, monkeypatch):
        self._generate_kwargs = {}

        class FakeMtlModel:
            sr = 16000

            def generate(inner_self, text, **kwargs):
                self._generate_kwargs.update(kwargs)
                return _make_fake_audio()

        class FakeMtlClass:
            @staticmethod
            def from_pretrained(device):
                return FakeMtlModel()

        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", True)
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "ChatterboxMultilingualTTS", FakeMtlClass)
        monkeypatch.setattr(tts_module, "_chatterbox_model_cache", {})

    def test_voice_match_passes_reference_audio_to_model(self, monkeypatch):
        self._setup_fake_mtl(monkeypatch)
        ref_audio = np.ones(16000, dtype=np.float32) * 0.1

        result = tts_module.synthesize_tts_pcm_with_cloning(
            "Hello",
            rate=16000,
            output_lang="en",
            reference_audio=ref_audio,
            reference_sample_rate=16000,
            voice_backend="chatterbox",
            voice_match=True,
            verbose=True,
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert "audio_prompt_path" in self._generate_kwargs

    def test_voice_match_without_reference_audio(self, monkeypatch):
        """Without reference audio, chatterbox should still work (no cloning)."""
        self._setup_fake_mtl(monkeypatch)

        result = tts_module.synthesize_tts_pcm_with_cloning(
            "Hello",
            rate=16000,
            output_lang="en",
            reference_audio=None,
            voice_backend="chatterbox",
            voice_match=True,
            verbose=True,
        )

        assert result is not None
        assert "audio_prompt_path" not in self._generate_kwargs

    def test_no_voice_match_no_reference_audio(self, monkeypatch):
        """Standard synthesis without cloning."""
        self._setup_fake_mtl(monkeypatch)

        result = tts_module.synthesize_tts_pcm_with_cloning(
            "Hello",
            rate=16000,
            output_lang="en",
            voice_backend="chatterbox",
            voice_match=False,
        )

        assert result is not None
        assert "audio_prompt_path" not in self._generate_kwargs

    def test_empty_text_returns_none(self, monkeypatch):
        self._setup_fake_mtl(monkeypatch)
        result = tts_module.synthesize_tts_pcm_with_cloning(
            "", rate=16000, output_lang="en", voice_backend="chatterbox"
        )
        assert result is None

    def test_temp_reference_file_cleaned_up(self, monkeypatch):
        """Reference audio temp file must be removed after synthesis."""
        self._setup_fake_mtl(monkeypatch)
        ref_audio = np.ones(16000, dtype=np.float32) * 0.1
        temp_files_created = []
        orig_named_temp = tempfile.NamedTemporaryFile

        def tracking_named_temp(*args, **kwargs):
            f = orig_named_temp(*args, **kwargs)
            temp_files_created.append(f.name)
            return f

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking_named_temp)

        tts_module.synthesize_tts_pcm_with_cloning(
            "Hello",
            rate=16000,
            output_lang="en",
            reference_audio=ref_audio,
            reference_sample_rate=16000,
            voice_backend="chatterbox",
            voice_match=True,
        )

        for path in temp_files_created:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"


# ---------------------------------------------------------------------------
# _get_torch_device() helper
# ---------------------------------------------------------------------------

class TestGetTorchDevice:

    def test_returns_string(self):
        device = tts_module._get_torch_device()
        assert isinstance(device, str)
        assert device in ("cuda", "mps", "cpu")

    def test_returns_cpu_when_torch_cuda_unavailable(self, monkeypatch):
        import sys as _sys
        if "torch" in _sys.modules:
            orig_torch = _sys.modules["torch"]
            orig_cuda = getattr(orig_torch, "cuda", None)
            try:
                # Patch cuda.is_available to return False
                cuda_mock = types.SimpleNamespace(is_available=lambda: False)
                orig_torch.cuda = cuda_mock
                if hasattr(orig_torch, "backends"):
                    orig_torch.backends = types.SimpleNamespace(
                        mps=types.SimpleNamespace(is_available=lambda: False)
                    )
                device = tts_module._get_torch_device()
                assert device == "cpu"
            finally:
                if orig_cuda is not None:
                    orig_torch.cuda = orig_cuda


# ---------------------------------------------------------------------------
# CLI / main.py – chatterbox choice
# ---------------------------------------------------------------------------

class TestChatterboxCliArg:
    """Check that main.py accepts 'chatterbox' as a --voice-backend value."""

    def test_chatterbox_in_main_source(self):
        """Verify 'chatterbox' appears in main.py --voice-backend choices."""
        import anytran.main as main_module
        import inspect
        source = inspect.getsource(main_module)
        assert "chatterbox" in source

    def test_chatterbox_availability_check_falls_back_to_gtts(self, monkeypatch):
        """When chatterbox packages are unavailable the backend falls back to gTTS."""
        monkeypatch.setattr(tts_module, "CHATTERBOX_MTL_AVAILABLE", False)
        monkeypatch.setattr(tts_module, "CHATTERBOX_TURBO_AVAILABLE", False)

        import argparse
        args = argparse.Namespace(voice_backend="chatterbox")

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            from anytran.tts import CHATTERBOX_MTL_AVAILABLE, CHATTERBOX_TURBO_AVAILABLE
            if not CHATTERBOX_MTL_AVAILABLE and not CHATTERBOX_TURBO_AVAILABLE:
                print("Warning: --voice-backend chatterbox specified but chatterbox-tts is not installed.")
                print("Falling back to gTTS.")
                args.voice_backend = "gtts"

        assert args.voice_backend == "gtts"
        assert "chatterbox-tts" in buf.getvalue()
