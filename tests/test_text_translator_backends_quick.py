from unittest.mock import MagicMock, patch

from tests.conftest import _real_text_translator_funcs


set_translategemma_config = _real_text_translator_funcs["set_translategemma_config"]
set_metanllb_config = _real_text_translator_funcs["set_metanllb_config"]
set_marianmt_config = _real_text_translator_funcs["set_marianmt_config"]
translate_text_libretranslate = _real_text_translator_funcs["translate_text_libretranslate"]


def test_backend_model_setters_update_globals():
    set_translategemma_config("google/model")
    set_metanllb_config("facebook/model")
    set_marianmt_config("Helsinki-NLP/opus-mt-en-fr")

    tg_globals = set_translategemma_config.__globals__
    nllb_globals = set_metanllb_config.__globals__
    marian_globals = set_marianmt_config.__globals__

    assert tg_globals["_translategemma_model_name"] == "google/model"
    assert nllb_globals["_metanllb_model_name"] == "facebook/model"
    assert marian_globals["_marianmt_model_name"] == "Helsinki-NLP/opus-mt-en-fr"


def test_translate_text_libretranslate_no_url_returns_none_verbose(capsys):
    fn_globals = translate_text_libretranslate.__globals__
    old_url = fn_globals["_libretranslate_url"]
    try:
        fn_globals["_libretranslate_url"] = None
        result = translate_text_libretranslate("hello", "en", "fr", verbose=True)
    finally:
        fn_globals["_libretranslate_url"] = old_url

    assert result is None
    assert "LibreTranslate URL not configured" in capsys.readouterr().out


def test_translate_text_libretranslate_success_request():
    fn_globals = translate_text_libretranslate.__globals__
    old_url = fn_globals["_libretranslate_url"]
    fn_globals["_libretranslate_url"] = "http://localhost:5000"
    try:
        response = MagicMock()
        response.json.return_value = {"translatedText": "bonjour"}
        response.raise_for_status.return_value = None

        with patch("requests.post", return_value=response) as post:
            out = translate_text_libretranslate("hello", "en", "fr", verbose=True)

        assert out == "bonjour"
        post.assert_called_once()
    finally:
        fn_globals["_libretranslate_url"] = old_url


def test_translate_text_libretranslate_request_exception_returns_none(capsys):
    fn_globals = translate_text_libretranslate.__globals__
    old_url = fn_globals["_libretranslate_url"]
    fn_globals["_libretranslate_url"] = "http://localhost:5000"
    try:
        with patch("requests.post", side_effect=RuntimeError("boom")):
            out = translate_text_libretranslate("hello", "en", "fr", verbose=True)
    finally:
        fn_globals["_libretranslate_url"] = old_url

    assert out is None
    assert "LibreTranslate translation failed" in capsys.readouterr().out
