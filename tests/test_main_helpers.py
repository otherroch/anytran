import os
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

import anytran.main as m


def _args(**overrides):
    base = dict(
        input=None,
        rtsp=None,
        from_output=False,
        youtube_url=None,
        web=False,
        scribe_text=None,
        scribe_voice=None,
        slate_text=None,
        slate_voice=None,
        mqtt_broker=None,
        youtube_api_key=None,
        web_ssl_cert=None,
        web_ssl_key=None,
        mqtt_topic_names=None,
        scribe_vad=False,
        voice_backend="gtts",
        input_lang=None,
        capture_voice=None,
        slate_backend="googletrans",
        libretranslate_url=None,
        slate_model=None,
        scribe_backend="faster-whisper",
        whispercpp_cli_detect_lang=False,
        whispercpp_model_dir="./models",
        scribe_model="medium",
        # NOTE: In tests we default no_auto_download to True to avoid external downloads;
        # this may intentionally differ from the application's default value.
        no_auto_download=True,
        whispercpp_bin="whisper-cli",
        whispercpp_threads=4,
        whisper_ctranslate2_device="auto",
        whisper_ctranslate2_device_index=None,
        whisper_ctranslate2_compute_type="default",
        verbose=False,
        chat_log="./chat",
        web_host="127.0.0.1",
    )
    base.update(overrides)
    return Namespace(**base)


def test_set_default_env_vars_windows_and_linux_defaults():
    with patch.dict(os.environ, {}, clear=True), patch("platform.system", return_value="Windows"):
        m._set_default_env_vars()
        assert os.environ["WHISPERCPP_BIN"].endswith("whisper-cli.exe")
        assert os.environ["WHISPERCPP_MODEL_NAME"] == "medium"

    with patch.dict(os.environ, {}, clear=True), patch("platform.system", return_value="Linux"):
        m._set_default_env_vars()
        assert os.environ["WHISPERCPP_BIN"].endswith("whisper-cli")
        assert os.environ["WHISPER_CTRANSLATE2_COMPUTE_TYPE"] == "default"


def test_validate_pipeline_args_missing_input_file_calls_parser_error():
    args = _args(input="missing.wav")
    parser = MagicMock()

    with patch("anytran.main.os.path.exists", return_value=False):
        m._validate_pipeline_args(args, parser)

    parser.error.assert_called_once()


def test_validate_pipeline_args_capture_voice_with_input_errors():
    args = _args(input="file.wav", capture_voice="cap.wav")
    parser = MagicMock()

    with patch("anytran.main.os.path.exists", return_value=True):
        m._validate_pipeline_args(args, parser)

    parser.error.assert_called_once()


def test_validate_pipeline_args_text_input_requires_non_auto_input_lang():
    args = _args(input="in.txt", input_lang="auto")
    parser = MagicMock()

    with patch("anytran.main.os.path.exists", return_value=True):
        m._validate_pipeline_args(args, parser)

    parser.error.assert_called_once()


def test_validate_pipeline_args_streaming_requires_output():
    args = _args(youtube_url="https://youtu.be/x", youtube_api_key="k")
    parser = MagicMock()
    m._validate_pipeline_args(args, parser)
    parser.error.assert_called_once()


def test_validate_pipeline_args_youtube_requires_api_key():
    args = _args(youtube_url="https://youtu.be/x", scribe_text="out.txt")
    parser = MagicMock()
    m._validate_pipeline_args(args, parser)
    parser.error.assert_called_once()


def test_validate_pipeline_args_web_ssl_pair_required():
    args = _args(web=True, web_ssl_cert="cert.pem", web_ssl_key=None, scribe_text="out.txt")
    parser = MagicMock()
    m._validate_pipeline_args(args, parser)
    parser.error.assert_called_once()


def test_validate_pipeline_args_rtsp_topic_count_must_match():
    args = _args(rtsp=["a", "b"], mqtt_topic_names=["only-one"], scribe_text="out.txt")
    parser = MagicMock()
    m._validate_pipeline_args(args, parser)
    parser.error.assert_called_once()


def test_validate_pipeline_args_piper_missing_falls_back_to_gtts():
    args = _args(voice_backend="piper", scribe_text="out.txt")
    parser = MagicMock()

    with patch("anytran.main.subprocess.run", side_effect=FileNotFoundError):
        m._validate_pipeline_args(args, parser)

    assert args.voice_backend == "gtts"


def test_configure_backends_text_input_skips_whisper_configuration():
    args = _args(input="in.txt", input_lang="en", slate_backend="libretranslate", libretranslate_url="http://lt")

    with patch("anytran.main.set_translation_backend") as set_trans, patch("anytran.main.set_libretranslate_config") as set_lt, patch(
        "anytran.main.set_whisper_backend"
    ) as set_wb:
        m._configure_backends(args)

    set_trans.assert_called_once_with("libretranslate")
    set_lt.assert_called_once_with("http://lt")
    set_wb.assert_not_called()


def test_configure_backends_whispercpp_cli_path():
    args = _args(scribe_backend="whispercpp-cli")

    with patch("anytran.main.set_translation_backend"), patch("anytran.main.set_whisper_backend") as set_wb, patch(
        "anytran.main.set_whispercpp_force_cli"
    ) as set_force, patch("anytran.main._configure_whispercpp") as cfg_cpp:
        m._configure_backends(args)

    set_wb.assert_called_once_with("whispercpp")
    set_force.assert_called_once_with(True)
    cfg_cpp.assert_called_once_with(args)


def test_configure_backends_whisper_ctranslate2_path():
    args = _args(scribe_backend="whisper-ctranslate2")

    with patch("anytran.main.set_translation_backend"), patch("anytran.main._configure_whisper_ctranslate2") as cfg_ct2:
        m._configure_backends(args)

    cfg_ct2.assert_called_once_with(args)


def test_configure_whispercpp_sets_config_with_existing_model(tmp_path):
    model = tmp_path / "ggml-medium.bin"
    model.write_bytes(b"x")

    args = _args(
        scribe_model=str(model),
        whispercpp_model_dir=str(tmp_path),
        whispercpp_bin="bin/whisper",
        no_auto_download=True,
    )

    with patch("anytran.main.resolve_path_with_fallback", side_effect=lambda p, r: p), patch(
        "anytran.main.set_whisper_cpp_config"
    ) as set_cfg:
        m._configure_whispercpp(args)

    set_cfg.assert_called_once()


def test_configure_whispercpp_missing_model_with_no_auto_download_exits():
    args = _args(
        scribe_model="missing-model",
        whispercpp_model_dir="./missing-models",
        whispercpp_bin="bin/whisper",
        no_auto_download=True,
    )

    with patch("anytran.main.os.path.exists", return_value=False), patch("anytran.main.resolve_path_with_fallback", side_effect=lambda p, r: p), patch(
        "anytran.main.sys.exit", side_effect=SystemExit
    ):
        with pytest.raises(SystemExit):
            m._configure_whispercpp(args)


def test_configure_whispercpp_auto_download_success_verbose(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    downloaded = tmp_path / "downloads" / "ggml-medium.bin"
    downloaded.parent.mkdir()
    downloaded.write_bytes(b"x")
    args = _args(
        scribe_model="medium",
        whispercpp_model_dir=str(models_dir),
        whispercpp_bin="bin/whisper",
        no_auto_download=False,
        verbose=True,
    )

    with patch("anytran.main.resolve_path_with_fallback", side_effect=lambda p, r: p), patch(
        "anytran.main.download_whisper_cpp_model", return_value=str(downloaded)
    ) as dl, patch("anytran.main.set_whisper_cpp_config") as set_cfg:
        m._configure_whispercpp(args)

    dl.assert_called_once()
    set_cfg.assert_called_once()


def test_configure_whispercpp_missing_after_config_exits():
    args = _args(
        scribe_model="medium",
        whispercpp_model_dir="./missing-models",
        whispercpp_bin="bin/whisper",
        no_auto_download=False,
    )

    with patch("anytran.main.os.path.exists", return_value=False), patch("anytran.main.resolve_path_with_fallback", side_effect=lambda p, r: p), patch(
        "anytran.main.download_whisper_cpp_model", return_value=None
    ), patch("anytran.main.sys.exit", side_effect=SystemExit), patch("anytran.main.set_whisper_cpp_config"):
        with pytest.raises(SystemExit):
            m._configure_whispercpp(args)


def test_configure_whisper_ctranslate2_rejects_missing_path_like_model():
    args = _args(scribe_model="./missing/model", whisper_ctranslate2_device="cpu")

    with patch("anytran.main.os.path.exists", return_value=False), patch("anytran.main.sys.exit", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            m._configure_whisper_ctranslate2(args)


def test_configure_whisper_ctranslate2_sets_config():
    args = _args(scribe_model="small", whisper_ctranslate2_device="cpu", whisper_ctranslate2_device_index=0)

    with patch("anytran.main.set_whisper_ctranslate2_config") as set_cfg:
        m._configure_whisper_ctranslate2(args)

    set_cfg.assert_called_once()


def test_ensure_chat_log_dir_creates_directory_and_updates_args(tmp_path):
    path = tmp_path / "chatdir"
    args = _args(chat_log=str(path), verbose=True)

    m._ensure_chat_log_dir(args)

    assert args.chat_log == str(path)
    assert path.exists()


def test_ensure_chat_log_dir_oserror_exits():
    args = _args(chat_log="bad")
    with patch("anytran.main.os.path.isdir", return_value=False), patch("anytran.main.os.makedirs", side_effect=OSError("nope")), patch(
        "anytran.main.sys.exit", side_effect=SystemExit
    ):
        with pytest.raises(SystemExit):
            m._ensure_chat_log_dir(args)


def test_generate_ssl_cert_if_needed_generates_defaults(tmp_path):
    args = _args(web_ssl_cert=None, web_ssl_key=None, web_host="localhost")

    with patch("anytran.main.os.getcwd", return_value=str(tmp_path)), patch("anytran.main.os.path.exists", return_value=False), patch(
        "anytran.main.generate_self_signed_cert"
    ) as gen:
        m._generate_ssl_cert_if_needed(args)

    assert args.web_ssl_cert.endswith("selfsigned.crt")
    assert args.web_ssl_key.endswith("selfsigned.key")
    gen.assert_called_once()


def test_generate_ssl_cert_if_needed_exception_exits(tmp_path):
    args = _args(web_ssl_cert=None, web_ssl_key=None, web_host="localhost")

    with patch("anytran.main.os.getcwd", return_value=str(tmp_path)), patch("anytran.main.os.path.exists", return_value=False), patch(
        "anytran.main.generate_self_signed_cert", side_effect=RuntimeError("fail")
    ), patch("anytran.main.sys.exit", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            m._generate_ssl_cert_if_needed(args)


def test_print_non_default_args_outputs_only_changed_values(capsys):
    args = _args(input_lang="fr", output_lang="en")
    defaults = {"input_lang": None, "output_lang": "en", "config": None, "genconfig": None}

    m._print_non_default_args(args, defaults)
    out = capsys.readouterr().out

    assert "Non-default settings:" in out
    assert "--input-lang = 'fr'" in out
    assert "--output-lang" not in out
