import ast
from pathlib import Path


VOICE_TABLE_APP = Path(__file__).resolve().parent.parent / "voice_table_app.py"


def _get_voices_by_language():
    tree = ast.parse(VOICE_TABLE_APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "VOICES_BY_LANGUAGE":
                    return ast.literal_eval(node.value)
    raise AssertionError("VOICES_BY_LANGUAGE not found")


def test_includes_french_plus_ten_additional_languages():
    voices_by_language = _get_voices_by_language()
    expected = {"fr", "en", "de", "es", "it", "pt", "ru", "pl", "nl", "ar", "zh"}
    assert expected.issubset(set(voices_by_language))


def test_languages_cli_argument_exists():
    text = VOICE_TABLE_APP.read_text(encoding="utf-8")
    assert "--languages" in text
