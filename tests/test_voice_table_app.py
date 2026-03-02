import json
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.voice_table_app as voice_table_app


def test_language_selection_supports_all_and_defaults():
    assert voice_table_app.get_selected_languages("fr,en") == ["fr", "en"]
    assert voice_table_app.get_selected_languages("") == ["fr"]
    assert voice_table_app.get_selected_languages("all") == ["all"]


def test_collect_voices_for_languages_accepts_family_and_full_code():
    catalog = {
        "fr_FR-gilles-low": {
            "language": {"family": "fr", "code": "fr_FR"},
            "files": {"fr/fr_FR/gilles/low/fr_FR-gilles-low.onnx": {}},
        },
        "ca_ES-upc_ona-x_low": {
            "language": {"family": "ca", "code": "ca_ES"},
            "files": {"ca/ca_ES/upc_ona/x_low/ca_ES-upc_ona-x_low.onnx": {}},
        },
    }

    voices, unknown = voice_table_app.collect_voices_for_languages(catalog, ["fr", "ca_es"])

    assert unknown == []
    assert {entry["onnx_file"] for entry in voices} == {
        "fr_FR-gilles-low.onnx",
        "ca_ES-upc_ona-x_low.onnx",
    }


def test_append_unique_entries_preserves_existing(tmp_path):
    output_file = tmp_path / "voice_table.json"
    output_file.write_text(
        json.dumps([{"onnx_file": "fr_FR-gilles-low.onnx", "pitch": 110, "gender": "male"}]),
        encoding="utf-8",
    )

    merged = voice_table_app.append_unique_entries(
        str(output_file),
        [
            {"onnx_file": "fr_FR-gilles-low.onnx", "pitch": 120, "gender": "male"},
            {"onnx_file": "fr_FR-siwis-low.onnx", "pitch": 190, "gender": "female"},
        ],
    )

    assert merged == [
        {"onnx_file": "fr_FR-gilles-low.onnx", "pitch": 110, "gender": "male"},
        {"onnx_file": "fr_FR-siwis-low.onnx", "pitch": 190, "gender": "female"},
    ]


def test_parse_args_uses_default_output():
    with mock.patch("sys.argv", ["voice_table_app.py"]):
        args = voice_table_app.parse_args()
    assert args.output == voice_table_app.DEFAULT_OUTPUT_FILE
