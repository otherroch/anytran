import json
import importlib.util
import builtins
from pathlib import Path
import numpy as np


def _load_voice_matcher_module():
    module_path = Path(__file__).resolve().parent.parent / "src" / "anytran" / "voice_matcher.py"
    spec = importlib.util.spec_from_file_location("voice_matcher_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_best_piper_voice_uses_json_table(tmp_path, monkeypatch):
    voice_matcher = _load_voice_matcher_module()
    table_path = tmp_path / "voice_table.json"
    table_path.write_text(
        json.dumps(
            [
                {"onnx_file": "en_US-foo-medium.onnx", "pitch": 120, "gender": "male"},
                {"onnx_file": "en_US-bar-medium.onnx", "pitch": 180, "gender": "female"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(voice_matcher, "VOICE_TABLE_JSON_PATH", table_path)
    voice_matcher._load_piper_voices.cache_clear()

    selected = voice_matcher.select_best_piper_voice(
        {"mean_pitch": 130, "gender": "male"},
        language="en",
    )

    assert selected == "en_US-foo-medium"


def test_select_best_piper_voice_uses_additional_attributes(tmp_path, monkeypatch):
    voice_matcher = _load_voice_matcher_module()
    table_path = tmp_path / "voice_table.json"
    table_path.write_text(
        json.dumps(
            [
                {
                    "onnx_file": "en_US-close-pitch-wrong-timbre.onnx",
                    "pitch": 151,
                    "gender": "male",
                    "zcr": 0.06,
                    "brightness": 3000,
                },
                {
                    "onnx_file": "en_US-slightly-farther-pitch-right-timbre.onnx",
                    "pitch": 156,
                    "gender": "male",
                    "zcr": 0.14,
                    "brightness": 1600,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(voice_matcher, "VOICE_TABLE_JSON_PATH", table_path)
    voice_matcher._load_piper_voices.cache_clear()

    selected = voice_matcher.select_best_piper_voice(
        {"mean_pitch": 150, "gender": "male", "zcr": 0.15, "brightness": 1500},
        language="en",
    )

    assert selected == "en_US-slightly-farther-pitch-right-timbre"


def test_voice_table_json_loaded_once(tmp_path, monkeypatch):
    voice_matcher = _load_voice_matcher_module()
    table_path = tmp_path / "voice_table.json"
    table_path.write_text(
        json.dumps(
            [{"onnx_file": "en_US-foo-medium.onnx", "pitch": 120, "gender": "male"}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(voice_matcher, "VOICE_TABLE_JSON_PATH", table_path)
    voice_matcher._load_piper_voices.cache_clear()

    original_open = builtins.open
    open_calls = {"count": 0}

    def counting_open(*args, **kwargs):
        if Path(args[0]) == table_path:
            open_calls["count"] += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    voice_matcher.select_best_piper_voice({"mean_pitch": 120, "gender": "male"}, language="en")
    voice_matcher.select_best_piper_voice({"mean_pitch": 121, "gender": "male"}, language="en")

    assert open_calls["count"] == 1


def test_gender_classification_handles_high_male_pitch(monkeypatch):
    voice_matcher = _load_voice_matcher_module()
    # Provide minimal librosa.feature functions to avoid heavy dependency behavior in test
    class _DummyLibrosa:
        @staticmethod
        def yin(*args, **kwargs):
            return np.array([205.0])

        class feature:
            @staticmethod
            def zero_crossing_rate(y):
                return [[0.05]]

            @staticmethod
            def spectral_centroid(y, sr):
                return [[1500.0]]

    monkeypatch.setattr(voice_matcher, "librosa", _DummyLibrosa)
    # Simulate a male-like voice around 205 Hz with low brightness so it should not be forced to female
    sample = np.sin(2 * np.pi * 205 * np.linspace(0, 1, 16000, endpoint=False)).astype(np.float32)
    features = voice_matcher.extract_voice_features(sample, sample_rate=16000, verbose=False)
    # Threshold adjustments should classify as male (male_mid) for ~205 Hz
    assert 200 <= features["mean_pitch"] <= 210
    assert features["gender"] == "male"
