import sys
from pathlib import Path
import numpy as np

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anytran import tts


def test_map_to_qwen_language():
    """Test language code mapping for Qwen3-TTS."""
    assert tts._map_to_qwen_language("en") == "English"
    assert tts._map_to_qwen_language("en-US") == "English"
    assert tts._map_to_qwen_language("zh") == "Chinese"
    assert tts._map_to_qwen_language("zh-CN") == "Chinese"
    assert tts._map_to_qwen_language("ja") == "Japanese"
    assert tts._map_to_qwen_language("ko") == "Korean"
    assert tts._map_to_qwen_language("de") == "German"
    assert tts._map_to_qwen_language("fr") == "French"
    assert tts._map_to_qwen_language("ru") == "Russian"
    assert tts._map_to_qwen_language("pt") == "Portuguese"
    assert tts._map_to_qwen_language("es") == "Spanish"
    assert tts._map_to_qwen_language("it") == "Italian"
    assert tts._map_to_qwen_language("unknown") == "Auto"
    assert tts._map_to_qwen_language(None) == "Auto"


def test_custom_tts_not_available_returns_false(tmp_path, monkeypatch):
    """Test that custom_tts returns False when qwen-tts is not installed."""
    monkeypatch.setattr(tts, "QWEN_TTS_AVAILABLE", False)
    
    output_wav = tmp_path / "output.wav"
    result = tts.custom_tts(
        "Hello world",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "en",
        str(output_wav),
        verbose=False
    )
    
    assert result is False


def test_custom_tts_with_mock_model(tmp_path, monkeypatch):
    """Test custom_tts with a mocked Qwen3TTSModel."""
    output_wav = tmp_path / "output.wav"
    
    # Mock torch
    class FakeTorch:
        class bfloat16:
            pass
    
    # Mock the model
    class FakeQwen3Model:
        def __init__(self):
            self.speakers = ["Ryan", "Vivian", "Serena"]
        
        def get_supported_speakers(self):
            return self.speakers
        
        def generate_custom_voice(self, text, language, speaker):
            # Return fake audio data: 1 second of silence at 24kHz
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            return [audio], sample_rate
        
        def generate_voice_clone(self, text, language, ref_audio, ref_text=None):
            # Return fake audio data: 1 second of silence at 24kHz
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            return [audio], sample_rate
        
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return FakeQwen3Model()
    
    monkeypatch.setattr(tts, "QWEN_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(tts, "torch", FakeTorch())
    monkeypatch.setattr(tts, "Qwen3TTSModel", FakeQwen3Model)
    monkeypatch.setattr(tts, "_custom_model_cache", {})
    
    # Test CustomVoice synthesis
    result = tts.custom_tts(
        "Hello world",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "en",
        str(output_wav),
        verbose=False
    )
    
    assert result is True
    assert output_wav.exists()


def test_custom_tts_with_voice_cloning(tmp_path, monkeypatch):
    """Test custom_tts with voice cloning using Base model."""
    output_wav = tmp_path / "output.wav"
    
    # Mock torch
    class FakeTorch:
        class bfloat16:
            pass
    
    # Mock the model
    class FakeQwen3Model:
        def generate_voice_clone(self, text, language, ref_audio, ref_text=None):
            # Return fake audio data
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            return [audio], sample_rate
        
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return FakeQwen3Model()
    
    monkeypatch.setattr(tts, "QWEN_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(tts, "torch", FakeTorch())
    monkeypatch.setattr(tts, "Qwen3TTSModel", FakeQwen3Model)
    monkeypatch.setattr(tts, "_custom_model_cache", {})
    
    # Create reference audio (1 second of random noise, clipped to valid audio range)
    ref_audio = np.clip(np.random.randn(16000), -1.0, 1.0).astype(np.float32)
    
    result = tts.custom_tts(
        "Hello world",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "en",
        str(output_wav),
        reference_audio=ref_audio,
        reference_text="This is the reference",
        verbose=False
    )
    
    assert result is True
    assert output_wav.exists()


def test_custom_tts_uses_cache(tmp_path, monkeypatch):
    """Test that custom_tts caches models."""
    output_wav = tmp_path / "output.wav"
    
    model_load_count = {"count": 0}
    
    # Mock torch
    class FakeTorch:
        class bfloat16:
            pass
    
    class FakeQwen3Model:
        def get_supported_speakers(self):
            return ["Ryan"]
        
        def generate_custom_voice(self, text, language, speaker):
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            return [audio], sample_rate
        
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            model_load_count["count"] += 1
            return FakeQwen3Model()
    
    monkeypatch.setattr(tts, "QWEN_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(tts, "torch", FakeTorch())
    monkeypatch.setattr(tts, "Qwen3TTSModel", FakeQwen3Model)
    monkeypatch.setattr(tts, "_custom_model_cache", {})
    
    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    
    # First call should load the model
    tts.custom_tts("Hello", model_name, "en", str(output_wav), verbose=False)
    assert model_load_count["count"] == 1
    
    # Second call should use cached model
    output_wav2 = tmp_path / "output2.wav"
    tts.custom_tts("World", model_name, "en", str(output_wav2), verbose=False)
    assert model_load_count["count"] == 1  # Should still be 1 (cached)


def test_synthesize_tts_pcm_with_custom_backend(monkeypatch):
    """Test synthesize_tts_pcm with custom backend."""
    
    # Mock torch
    class FakeTorch:
        class bfloat16:
            pass
    
    class FakeQwen3Model:
        def get_supported_speakers(self):
            return ["Ryan"]
        
        def generate_custom_voice(self, text, language, speaker):
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            return [audio], sample_rate
        
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return FakeQwen3Model()
    
    monkeypatch.setattr(tts, "QWEN_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(tts, "torch", FakeTorch())
    monkeypatch.setattr(tts, "Qwen3TTSModel", FakeQwen3Model)
    monkeypatch.setattr(tts, "_custom_model_cache", {})
    
    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="custom",
        voice_model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        verbose=False
    )
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16


def test_synthesize_tts_pcm_with_cloning_custom_backend(monkeypatch):
    """Test synthesize_tts_pcm_with_cloning with custom backend and voice matching."""
    
    # Mock torch
    class FakeTorch:
        class bfloat16:
            pass
    
    ref_text_received = []
    
    class FakeQwen3Model:
        def generate_voice_clone(self, text, language, ref_audio, ref_text=None):
            ref_text_received.append(ref_text)
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            return [audio], sample_rate
        
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return FakeQwen3Model()
    
    monkeypatch.setattr(tts, "QWEN_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(tts, "torch", FakeTorch())
    monkeypatch.setattr(tts, "Qwen3TTSModel", FakeQwen3Model)
    monkeypatch.setattr(tts, "_custom_model_cache", {})
    
    # Create reference audio (properly scaled int16 audio)
    ref_audio = (np.clip(np.random.randn(16000), -1.0, 1.0) * 32767).astype(np.int16)
    
    # Test with reference_text
    result = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=ref_audio,
        reference_sample_rate=16000,
        reference_text="This is the reference text",
        voice_backend="custom",
        voice_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        voice_match=True,
        verbose=False
    )
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    assert len(ref_text_received) == 1
    assert ref_text_received[0] == "This is the reference text"


def test_custom_backend_replaces_piper_model(monkeypatch):
    """Test that Piper model names are replaced with Qwen3-TTS models."""
    
    # Mock torch
    class FakeTorch:
        class bfloat16:
            pass
    
    model_loads = []
    
    class FakeQwen3Model:
        def get_supported_speakers(self):
            return ["Ryan"]
        
        def generate_custom_voice(self, text, language, speaker):
            sample_rate = 24000
            audio = np.zeros(sample_rate, dtype=np.float32)
            return [audio], sample_rate
        
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            model_loads.append(model_name)
            if "Qwen" not in model_name:
                raise ValueError(f"{model_name} is not a valid model")
            return FakeQwen3Model()
    
    monkeypatch.setattr(tts, "QWEN_TTS_AVAILABLE", True)
    monkeypatch.setattr(tts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(tts, "torch", FakeTorch())
    monkeypatch.setattr(tts, "Qwen3TTSModel", FakeQwen3Model)
    monkeypatch.setattr(tts, "_custom_model_cache", {})
    
    # Test that Piper model is replaced
    result = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="custom",
        voice_model="en_US-lessac-medium",  # Piper model
        verbose=False
    )
    
    assert result is not None
    assert len(model_loads) == 1
    # Should have loaded a Qwen model, not the Piper model
    assert "Qwen" in model_loads[0]
    assert "en_US-lessac-medium" not in model_loads[0]
