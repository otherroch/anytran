import json
import sys
from pathlib import Path
import wave

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anytran import tts


def test_find_piper_config_path_prefers_matching_json(tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"onnx")
    config_path = tmp_path / "voice.onnx.json"
    config_path.write_text("{}", encoding="utf-8")

    detected = tts._find_piper_config_path(str(model_path))

    assert detected == str(config_path)


def test_resolve_piper_sample_rate_reads_config(tmp_path):
    config_path = tmp_path / "voice.onnx.json"
    config_path.write_text(json.dumps({"audio": {"sample_rate": 44100}}), encoding="utf-8")

    class DummyVoice:
        pass

    resolved = tts._resolve_piper_sample_rate(DummyVoice(), str(config_path))

    assert resolved == 44100


def test_piper_tts_uses_config_and_sets_wav_header(monkeypatch, tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"onnx")
    config_path = tmp_path / "voice.onnx.json"
    config_path.write_text(json.dumps({"audio": {"sample_rate": 32000}}), encoding="utf-8")
    output_wav = tmp_path / "out.wav"

    load_kwargs = {}
    synth_calls = {}

    class FakeVoice:
        def __init__(self):
            self.sample_rate = 0

        def synthesize_wav(self, text, wav_file):
            synth_calls["text"] = text
            wav_file.writeframes(b"data")

    class FakePiperVoice:
        @staticmethod
        def load(path, **kwargs):
            load_kwargs["path"] = path
            load_kwargs["config_path"] = kwargs.get("config_path")
            return FakeVoice()

    class FakeWaveFile:
        def __init__(self):
            self.params = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def setnchannels(self, value):
            self.params["channels"] = value

        def setsampwidth(self, value):
            self.params["width"] = value

        def setframerate(self, value):
            self.params["rate"] = value

        def writeframes(self, data):
            self.params["frames"] = data

    fake_wave_file = FakeWaveFile()

    def fake_wave_open(path, mode):
        assert path == str(output_wav)
        assert mode == "wb"
        return fake_wave_file

    monkeypatch.setattr(tts, "PIPER_PYTHON_AVAILABLE", True)
    monkeypatch.setattr(tts, "PiperVoice", FakePiperVoice)
    monkeypatch.setattr(tts, "_piper_voice_cache", {})
    monkeypatch.setattr(wave, "open", fake_wave_open)

    success = tts.piper_tts("hello", str(model_path), str(output_wav), verbose=False)

    assert success is True
    assert load_kwargs["path"] == str(model_path)
    assert load_kwargs["config_path"] == str(config_path)
    assert synth_calls["text"] == "hello"
    assert fake_wave_file.params["channels"] == 1
    assert fake_wave_file.params["width"] == 2
    assert fake_wave_file.params["rate"] == 32000
    assert fake_wave_file.params["frames"] == b"data"
