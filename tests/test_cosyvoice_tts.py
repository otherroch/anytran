import sys
from pathlib import Path
import tempfile

import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anytran import tts


def test_ensure_cosyvoice_available_when_not_installed(monkeypatch):
    """Test that _ensure_cosyvoice_available returns False when CosyVoice is not available."""
    monkeypatch.setattr(tts, "COSYVOICE_AVAILABLE", False)
    
    result = tts._ensure_cosyvoice_available(verbose=False)
    
    assert result is False


def test_ensure_cosyvoice_available_when_installed(monkeypatch):
    """Test that _ensure_cosyvoice_available returns True when CosyVoice is available."""
    monkeypatch.setattr(tts, "COSYVOICE_AVAILABLE", True)
    
    result = tts._ensure_cosyvoice_available(verbose=False)
    
    assert result is True


def test_cosyvoice_tts_basic_synthesis(monkeypatch, tmp_path):
    """Test basic CosyVoice TTS synthesis without reference audio."""
    output_wav = tmp_path / "output.wav"
    
    # Mock audio output - 1 second at 22050 Hz
    sample_rate = 22050
    duration_seconds = 1.0
    mock_audio = np.random.randn(int(sample_rate * duration_seconds))
    
    class FakeCosyVoice:
        def __init__(self, model_path):
            self.model_path = model_path
        
        def inference_sft(self, text):
            return mock_audio
    
    # Track what model was loaded
    load_calls = []
    
    def fake_cosyvoice_init(model_path):
        load_calls.append(model_path)
        return FakeCosyVoice(model_path)
    
    # Mock soundfile write
    write_calls = []
    def fake_sf_write(path, data, samplerate):
        write_calls.append({
            "path": path,
            "data": data,
            "samplerate": samplerate
        })
    
    monkeypatch.setattr(tts, "COSYVOICE_AVAILABLE", True)
    monkeypatch.setattr(tts, "CosyVoice", fake_cosyvoice_init)
    monkeypatch.setattr(tts, "_cosyvoice_model_cache", {})
    
    import soundfile
    monkeypatch.setattr(soundfile, "write", fake_sf_write)
    
    # Test synthesis
    success = tts.cosyvoice_tts(
        "Hello world", 
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        str(output_wav),
        verbose=False
    )
    
    assert success is True
    assert len(write_calls) == 1
    assert write_calls[0]["path"] == str(output_wav)
    assert write_calls[0]["samplerate"] == 22050
    assert np.array_equal(write_calls[0]["data"], mock_audio)


def test_cosyvoice_tts_with_reference_audio(monkeypatch, tmp_path):
    """Test CosyVoice TTS synthesis with reference audio for voice cloning."""
    output_wav = tmp_path / "output.wav"
    reference_wav = tmp_path / "reference.wav"
    reference_wav.write_bytes(b"fake audio")
    
    # Mock audio output
    mock_audio = np.random.randn(22050)
    
    class FakeCosyVoice:
        def __init__(self, model_path):
            self.model_path = model_path
        
        def inference_zero_shot(self, text, reference_path):
            self.last_reference = reference_path
            return mock_audio
    
    fake_model = None
    
    def fake_cosyvoice_init(model_path):
        nonlocal fake_model
        fake_model = FakeCosyVoice(model_path)
        return fake_model
    
    # Mock soundfile write
    write_calls = []
    def fake_sf_write(path, data, samplerate):
        write_calls.append({
            "path": path,
            "data": data,
            "samplerate": samplerate
        })
    
    monkeypatch.setattr(tts, "COSYVOICE_AVAILABLE", True)
    monkeypatch.setattr(tts, "CosyVoice", fake_cosyvoice_init)
    monkeypatch.setattr(tts, "_cosyvoice_model_cache", {})
    
    import soundfile
    monkeypatch.setattr(soundfile, "write", fake_sf_write)
    
    # Test synthesis with reference audio
    success = tts.cosyvoice_tts(
        "Hello world",
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        str(output_wav),
        reference_audio_path=str(reference_wav),
        verbose=False
    )
    
    assert success is True
    assert fake_model.last_reference == str(reference_wav)
    assert len(write_calls) == 1


def test_cosyvoice_tts_model_caching(monkeypatch, tmp_path):
    """Test that CosyVoice models are cached properly."""
    output_wav = tmp_path / "output.wav"
    
    mock_audio = np.random.randn(22050)
    
    class FakeCosyVoice:
        def __init__(self, model_path):
            self.model_path = model_path
        
        def inference_sft(self, text):
            return mock_audio
    
    load_count = [0]
    
    def fake_cosyvoice_init(model_path):
        load_count[0] += 1
        return FakeCosyVoice(model_path)
    
    # Mock soundfile write
    def fake_sf_write(path, data, samplerate):
        pass
    
    monkeypatch.setattr(tts, "COSYVOICE_AVAILABLE", True)
    monkeypatch.setattr(tts, "CosyVoice", fake_cosyvoice_init)
    monkeypatch.setattr(tts, "_cosyvoice_model_cache", {})
    
    import soundfile
    monkeypatch.setattr(soundfile, "write", fake_sf_write)
    
    model_name = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    
    # First call - should load model
    tts.cosyvoice_tts("Hello", model_name, str(output_wav), verbose=False)
    assert load_count[0] == 1
    
    # Second call - should use cached model
    tts.cosyvoice_tts("World", model_name, str(output_wav), verbose=False)
    assert load_count[0] == 1  # Still 1, model was cached


def test_synthesize_tts_pcm_with_cosyvoice(monkeypatch, tmp_path):
    """Test synthesize_tts_pcm with cosyvoice backend."""
    
    # Create mock audio
    mock_audio = np.random.randn(22050)
    
    class FakeCosyVoice:
        def __init__(self, model_path):
            pass
        
        def inference_sft(self, text):
            return mock_audio
    
    def fake_cosyvoice_init(model_path):
        return FakeCosyVoice(model_path)
    
    # Mock soundfile operations
    def fake_sf_write(path, data, samplerate):
        pass
    
    def fake_sf_read(path):
        return mock_audio, 22050
    
    monkeypatch.setattr(tts, "COSYVOICE_AVAILABLE", True)
    monkeypatch.setattr(tts, "CosyVoice", fake_cosyvoice_init)
    monkeypatch.setattr(tts, "_cosyvoice_model_cache", {})
    
    import soundfile
    monkeypatch.setattr(soundfile, "write", fake_sf_write)
    monkeypatch.setattr(soundfile, "read", fake_sf_read)
    
    # Test synthesis
    pcm = tts.synthesize_tts_pcm(
        "Hello world",
        rate=16000,
        output_lang="en",
        voice_backend="cosyvoice",
        voice_model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        verbose=False
    )
    
    assert pcm is not None
    assert isinstance(pcm, np.ndarray)
    assert pcm.dtype == np.int16


def test_synthesize_tts_pcm_with_cloning_cosyvoice(monkeypatch, tmp_path):
    """Test synthesize_tts_pcm_with_cloning with cosyvoice and voice matching."""
    
    # Create mock audio and reference
    mock_audio = np.random.randn(22050)
    reference_audio = np.random.randn(16000)
    
    class FakeCosyVoice:
        def __init__(self, model_path):
            self.last_reference = None
        
        def inference_zero_shot(self, text, reference_path):
            self.last_reference = reference_path
            return mock_audio
    
    fake_model = None
    
    def fake_cosyvoice_init(model_path):
        nonlocal fake_model
        fake_model = FakeCosyVoice(model_path)
        return fake_model
    
    # Mock soundfile operations
    sf_writes = []
    def fake_sf_write(path, data, samplerate):
        sf_writes.append((path, data, samplerate))
    
    def fake_sf_read(path):
        return mock_audio, 22050
    
    monkeypatch.setattr(tts, "COSYVOICE_AVAILABLE", True)
    monkeypatch.setattr(tts, "CosyVoice", fake_cosyvoice_init)
    monkeypatch.setattr(tts, "_cosyvoice_model_cache", {})
    
    import soundfile
    monkeypatch.setattr(soundfile, "write", fake_sf_write)
    monkeypatch.setattr(soundfile, "read", fake_sf_read)
    
    # Test synthesis with voice cloning
    pcm = tts.synthesize_tts_pcm_with_cloning(
        "Hello world",
        rate=16000,
        output_lang="en",
        reference_audio=reference_audio,
        reference_sample_rate=16000,
        voice_backend="cosyvoice",
        voice_model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        voice_match=True,
        verbose=False
    )
    
    assert pcm is not None
    assert isinstance(pcm, np.ndarray)
    assert pcm.dtype == np.int16
    # Verify reference audio was used
    assert fake_model.last_reference is not None
    # First write should be the reference audio
    assert len(sf_writes) >= 1
