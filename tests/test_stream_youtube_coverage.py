import types
from unittest.mock import MagicMock, patch

import numpy as np

import anytran.stream_youtube as sy


class _Queue:
    def __init__(self):
        self.items = []

    def full(self):
        return False

    def get_nowait(self):
        if self.items:
            return self.items.pop(0)
        raise Exception("empty")

    def put(self, item, timeout=None):
        self.items.append(item)


def test_validate_youtube_video_missing_inputs_returns_none():
    assert sy.validate_youtube_video(None, "abc") is None
    assert sy.validate_youtube_video("k", None) is None


def test_validate_youtube_video_success_parses_payload():
    payload = (
        '{"items":[{"snippet":{"title":"t"},"contentDetails":{"duration":"PT3M"},'
        '"liveStreamingDetails":{"actualStartTime":"now"}}]}'
    )
    response = MagicMock()
    response.read.return_value = payload.encode("utf-8")

    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False

    with patch("anytran.stream_youtube.urllib.request.urlopen", return_value=cm) as urlopen:
        data = sy.validate_youtube_video("k", "id123", verbose=True)

    assert data["snippet"]["title"] == "t"
    assert data["contentDetails"]["duration"] == "PT3M"
    assert data["liveStreamingDetails"]["actualStartTime"] == "now"
    assert urlopen.called


def test_validate_youtube_video_empty_items_returns_none():
    response = MagicMock()
    response.read.return_value = b'{"items":[]}'
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False

    with patch("anytran.stream_youtube.urllib.request.urlopen", return_value=cm):
        assert sy.validate_youtube_video("k", "id123") is None


def test_extract_youtube_video_id_live_url():
    assert sy.extract_youtube_video_id("https://www.youtube.com/live/abc123xyz") == "abc123xyz"


def test_get_youtube_audio_stream_url_import_error_returns_none(capsys):
    with patch.dict("sys.modules", {"yt_dlp": None}):
        out = sy.get_youtube_audio_stream_url("https://youtu.be/x")
    assert out is None
    printed = capsys.readouterr().out
    assert "yt-dlp not installed or import failed" in printed


def test_get_youtube_audio_stream_url_uses_direct_info_url():
    ydl_instance = MagicMock()
    ydl_instance.extract_info.return_value = {"url": "https://audio.example/direct"}

    ydl_cls = MagicMock()
    ydl_cm = MagicMock()
    ydl_cm.__enter__.return_value = ydl_instance
    ydl_cm.__exit__.return_value = False
    ydl_cls.return_value = ydl_cm

    mock_mod = types.SimpleNamespace(YoutubeDL=ydl_cls)
    with patch.dict("sys.modules", {"yt_dlp": mock_mod}):
        url = sy.get_youtube_audio_stream_url("https://youtu.be/x", js_runtime="node:C:/node.exe", remote_components="all")

    assert url == "https://audio.example/direct"
    opts = ydl_cls.call_args.args[0]
    assert opts["js_runtimes"] == {"node": {"path": "C:/node.exe"}}
    assert opts["remote_components"] == ["all"]


def test_get_youtube_audio_stream_url_uses_audio_only_fallback():
    ydl_instance = MagicMock()
    ydl_instance.extract_info.return_value = {
        "formats": [
            {"vcodec": "avc1", "url": "https://video"},
            {"vcodec": "none", "url": "https://audio1"},
            {"vcodec": "none", "url": "https://audio2"},
        ]
    }

    ydl_cls = MagicMock()
    ydl_cm = MagicMock()
    ydl_cm.__enter__.return_value = ydl_instance
    ydl_cm.__exit__.return_value = False
    ydl_cls.return_value = ydl_cm

    mock_mod = types.SimpleNamespace(YoutubeDL=ydl_cls)
    with patch.dict("sys.modules", {"yt_dlp": mock_mod}):
        url = sy.get_youtube_audio_stream_url("https://youtu.be/x")

    assert url == "https://audio2"


def test_get_youtube_audio_stream_url_general_exception_returns_none(capsys):
    ydl_cls = MagicMock(side_effect=RuntimeError("boom"))
    mock_mod = types.SimpleNamespace(YoutubeDL=ydl_cls)
    with patch.dict("sys.modules", {"yt_dlp": mock_mod}):
        url = sy.get_youtube_audio_stream_url("https://youtu.be/x")
    assert url is None
    assert "yt-dlp failed to resolve audio stream" in capsys.readouterr().out


def test_stream_youtube_audio_no_url_prints_and_returns(capsys):
    q = _Queue()
    sy.stream_youtube_audio(lambda: None, q)
    out = capsys.readouterr().out
    assert "Unable to resolve YouTube audio stream URL" in out


def test_stream_youtube_audio_import_error(capsys):
    q = _Queue()
    with patch.dict("sys.modules", {"av": None}):
        sy.stream_youtube_audio(lambda: "https://audio.example", q)
    out = capsys.readouterr().out
    assert "PyAV not installed" in out


def test_stream_youtube_audio_decodes_one_frame_and_stops():
    q = _Queue()

    class Stop:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 2

    stop = Stop()

    frame = MagicMock()
    resampled_frame = MagicMock()
    resampled_frame.to_ndarray.return_value = np.array([[0, 32767]], dtype=np.int16)

    container = MagicMock()
    container.streams = [types.SimpleNamespace(type="audio")]
    container.decode.return_value = [frame]

    resampler_inst = MagicMock()
    resampler_inst.resample.return_value = [resampled_frame]

    av_mod = types.SimpleNamespace(
        open=MagicMock(return_value=container),
        time_base=1,
        audio=types.SimpleNamespace(resampler=types.SimpleNamespace(AudioResampler=MagicMock(return_value=resampler_inst))),
    )

    with patch.dict("sys.modules", {"av": av_mod}):
        sy.stream_youtube_audio(lambda: "https://audio.example", q, stop_event=stop, expected_duration=0.001, verbose=True)

    assert len(q.items) >= 1
    assert isinstance(q.items[0], np.ndarray)
    container.close.assert_called()


def test_stream_youtube_audio_no_audio_stream_returns(capsys):
    q = _Queue()

    container = MagicMock()
    container.streams = [types.SimpleNamespace(type="video")]

    av_mod = types.SimpleNamespace(
        open=MagicMock(return_value=container),
        time_base=1,
        audio=types.SimpleNamespace(resampler=types.SimpleNamespace(AudioResampler=MagicMock())),
    )

    with patch.dict("sys.modules", {"av": av_mod}):
        sy.stream_youtube_audio(lambda: "https://audio.example", q)

    assert "No audio stream found" in capsys.readouterr().out


def test_stream_youtube_audio_decode_error_then_retries_to_limit(capsys):
    q = _Queue()

    class Stop:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return False

    stop = Stop()

    container = MagicMock()
    container.streams = [types.SimpleNamespace(type="audio")]
    container.decode.side_effect = RuntimeError("decode failed")

    av_mod = types.SimpleNamespace(
        open=MagicMock(return_value=container),
        time_base=1,
        audio=types.SimpleNamespace(resampler=types.SimpleNamespace(AudioResampler=MagicMock(return_value=MagicMock()))),
    )

    with patch.dict("sys.modules", {"av": av_mod}):
        sy.stream_youtube_audio(lambda: "https://audio.example", q, stop_event=stop, max_retries=0, verbose=True)

    out = capsys.readouterr().out
    assert "YouTube stream decode error" in out
    assert "Max YouTube reconnect attempts reached" in out
