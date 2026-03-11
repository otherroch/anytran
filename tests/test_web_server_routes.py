import asyncio
import json
from unittest.mock import MagicMock, patch

from tests.conftest import _real_web_server_funcs


run_web_server = _real_web_server_funcs["run_web_server"]


def _capture_app_and_run_server(extra_run_kwargs=None):
    captured = {}
    server_mock = MagicMock()

    def config_side_effect(app, **kwargs):
        captured["app"] = app
        return MagicMock()

    with patch("uvicorn.Config", side_effect=config_side_effect), patch("uvicorn.Server", return_value=server_mock), patch(
        "signal.signal"
    ), patch("anytran.web_server.init_mqtt"), patch("anytran.web_server.get_whisper_backend", return_value="whispercpp"):
        kwargs = dict(
            input_lang="auto",
            output_lang="en",
            magnitude_threshold=0.02,
            model=None,
            host="127.0.0.1",
            port=8899,
            verbose=False,
            mqtt_broker=None,
            mqtt_port=1883,
            mqtt_username=None,
            mqtt_password=None,
            mqtt_topic="translation",
            ssl_certfile=None,
            ssl_keyfile=None,
            window_seconds=5.0,
            overlap_seconds=0.0,
            dedup=False,
            keep_temp=False,
            scribe_vad=False,
            timers=False,
            timers_all=False,
            scribe_backend="none",
            slate_backend="none",
            lang_prefix=False,
            voice_backend="auto",
            voice_model=None,
            voice_match=False,
            capture_voice_path=None,
        )
        if extra_run_kwargs:
            kwargs.update(extra_run_kwargs)
        run_web_server(**kwargs)

    return captured["app"], server_mock


def test_run_web_server_starts_uvicorn_server_run_called():
    _, server_mock = _capture_app_and_run_server()
    server_mock.run.assert_called_once()


def test_index_route_returns_html_page():
    app, _ = _capture_app_and_run_server()

    route = next(r for r in app.routes if getattr(r, "path", None) == "/")
    resp = route.endpoint()
    assert resp.status_code == 200
    assert "<!doctype html>" in resp.body.decode("utf-8").lower()


class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []

    async def accept(self):
        return None

    async def receive(self):
        if not self._messages:
            from fastapi import WebSocketDisconnect

            raise WebSocketDisconnect(code=1000)
        return self._messages.pop(0)

    async def send_text(self, text):
        self.sent.append(text)


def test_websocket_accepts_config_messages_and_disconnects_cleanly():
    app, _ = _capture_app_and_run_server()

    with patch("anytran.web_server.process_audio_chunk", return_value=None):
        ws_route = next(r for r in app.routes if getattr(r, "path", None) == "/ws")
        ws = _FakeWebSocket(
            [
                {"text": json.dumps({"type": "config", "input_lang": "fr"})},
                {"text": json.dumps({"type": "config", "output_lang": "es"})},
                {"text": json.dumps({"type": "config", "langswap": True})},
            ]
        )
        asyncio.run(ws_route.endpoint(ws))

    assert any("input_lang set to fr" in msg for msg in ws.sent)
    assert any("output_lang set to es" in msg for msg in ws.sent)
    assert any("LangSwap enabled" in msg for msg in ws.sent)


def test_websocket_binary_audio_sends_translation_payload():
    app, _ = _capture_app_and_run_server(extra_run_kwargs={"voice_backend": "piper", "verbose": True})

    translated = {"output": "bonjour", "final_lang": "fr"}
    with patch("anytran.web_server.process_audio_chunk", return_value=translated):
        ws_route = next(r for r in app.routes if getattr(r, "path", None) == "/ws")
        ws = _FakeWebSocket([{"bytes": (b"\x00\x00") * 90000}])
        asyncio.run(ws_route.endpoint(ws))

    payloads = [json.loads(msg) for msg in ws.sent if isinstance(msg, str) and msg.startswith("{")]
    translation = next(p for p in payloads if p.get("type") == "translation")
    assert translation["text"] == "bonjour"
    assert translation["lang"] == "fr-FR"
