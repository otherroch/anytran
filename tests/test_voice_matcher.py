import json
import importlib.util
from pathlib import Path


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
