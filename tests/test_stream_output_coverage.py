from unittest.mock import MagicMock, patch

import numpy as np

from tests.conftest import _real_stream_output_funcs


get_wasapi_loopback_device_info = _real_stream_output_funcs["get_wasapi_loopback_device_info"]
list_wasapi_loopback_devices = _real_stream_output_funcs["list_wasapi_loopback_devices"]
stream_output_audio = _real_stream_output_funcs["stream_output_audio"]


class _Queue:
    def __init__(self, full_first=False):
        self.items = []
        self._full_first = full_first

    def full(self):
        if self._full_first:
            self._full_first = False
            return True
        return False

    def get_nowait(self):
        if self.items:
            return self.items.pop(0)
        raise Exception("empty")

    def put(self, item):
        self.items.append(item)


class _StopAfterOne:
    def __init__(self):
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > 1


def _make_pyaudio_module(devices, default_output=None, wasapi_index=1):
    p = MagicMock()
    p.get_host_api_info_by_type.return_value = {"index": wasapi_index}
    p.get_default_output_device_info.return_value = default_output or {}
    p.get_device_count.return_value = len(devices)
    p.get_device_info_by_index.side_effect = lambda idx: devices[idx]

    mod = MagicMock()
    mod.paWASAPI = 1234
    mod.paInt16 = 8
    mod.PyAudio.return_value = p
    return mod, p


def test_get_wasapi_loopback_device_info_non_windows_returns_none():
    with patch("anytran.stream_output.sys.platform", "linux"):
        assert get_wasapi_loopback_device_info() is None


def test_get_wasapi_loopback_device_info_prefers_name_match():
    devices = [
        {"index": 1, "hostApi": 1, "maxInputChannels": 2, "name": "Speaker (loopback)", "isLoopbackDevice": True},
        {"index": 2, "hostApi": 1, "maxInputChannels": 2, "name": "Headset loopback", "isLoopbackDevice": True},
    ]
    mod, p = _make_pyaudio_module(devices, default_output={"name": "Speaker"})

    with patch("anytran.stream_output.sys.platform", "win32"), patch.dict("sys.modules", {"pyaudiowpatch": mod}):
        info = get_wasapi_loopback_device_info(preferred_name="headset")

    assert info["index"] == 2
    p.terminate.assert_called_once()


def test_get_wasapi_loopback_device_info_returns_none_when_no_loopback():
    devices = [{"index": 1, "hostApi": 1, "maxInputChannels": 0, "name": "Mic", "isLoopbackDevice": False}]
    mod, _ = _make_pyaudio_module(devices)

    with patch("anytran.stream_output.sys.platform", "win32"), patch.dict("sys.modules", {"pyaudiowpatch": mod}):
        info = get_wasapi_loopback_device_info()

    assert info is None


def test_list_wasapi_loopback_devices_non_windows_prints_message(capsys):
    with patch("anytran.stream_output.sys.platform", "darwin"):
        list_wasapi_loopback_devices()
    out = capsys.readouterr().out
    assert "only supported on Windows" in out


def test_list_wasapi_loopback_devices_prints_devices(capsys):
    devices = [{"index": 7, "hostApi": 1, "maxInputChannels": 2, "name": "Realtek loopback", "isLoopbackDevice": True}]
    mod, p = _make_pyaudio_module(devices, default_output={"name": "Realtek"})

    with patch("anytran.stream_output.sys.platform", "win32"), patch.dict("sys.modules", {"pyaudiowpatch": mod}):
        list_wasapi_loopback_devices()

    out = capsys.readouterr().out
    assert "WASAPI loopback devices:" in out
    assert "Realtek loopback" in out
    assert "default" in out
    p.terminate.assert_called_once()


def test_stream_output_audio_non_windows_exits(capsys):
    q = _Queue()
    with patch("anytran.stream_output.sys.platform", "linux"):
        stream_output_audio(q, {"index": 1})
    out = capsys.readouterr().out
    assert "only supported on Windows" in out


def test_stream_output_audio_no_device_exits(capsys):
    q = _Queue()
    with patch("anytran.stream_output.sys.platform", "win32"):
        stream_output_audio(q, None)
    out = capsys.readouterr().out
    assert "No WASAPI loopback device found" in out


def test_stream_output_audio_reads_resamples_and_drops_oldest():
    p = MagicMock()
    stream = MagicMock()
    p.open.return_value = stream

    # 4 stereo sample pairs (8 individual int16 values); function averages channels to mono.
    raw = np.array([0, 32767, 0, 32767, 0, 32767, 0, 32767], dtype=np.int16).tobytes()
    stream.read.return_value = raw

    mod = MagicMock()
    mod.paInt16 = 8
    mod.PyAudio.return_value = p

    q = _Queue(full_first=True)
    q.items.append(np.array([1.0], dtype=np.float32))
    stop = _StopAfterOne()

    with patch("anytran.stream_output.sys.platform", "win32"), patch.dict("sys.modules", {"pyaudiowpatch": mod}), patch(
        "anytran.stream_output.librosa.resample", return_value=np.array([0.5, 0.5], dtype=np.float32)
    ) as mock_resample:
        stream_output_audio(
            q,
            {"index": 3, "defaultSampleRate": 48000, "maxInputChannels": 2, "name": "Loopback"},
            target_rate=16000,
            stop_event=stop,
            verbose=True,
        )

    assert len(q.items) == 1
    assert np.allclose(q.items[0], np.array([0.5, 0.5], dtype=np.float32))
    mock_resample.assert_called_once()
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    p.terminate.assert_called_once()
