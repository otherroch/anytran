import importlib
import sys
import time
import numpy as np
import pytest


@pytest.fixture
def runner_modules():
    module_names = [
        "anytran.runners.run_file_input",
        "anytran.runners.run_realtime_output",
        "anytran.runners.run_realtime_youtube",
        "anytran.runners.run_realtime_rtsp",
        "anytran.runners.run_multi_rtsp",
        "anytran.runners",
        "anytran.stream_output",
        "anytran.stream_youtube",
    ]
    original_module_states = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    modules = (
        importlib.import_module("anytran.runners.run_file_input"),
        importlib.import_module("anytran.runners.run_realtime_output"),
        importlib.import_module("anytran.runners.run_realtime_youtube"),
        importlib.import_module("anytran.runners.run_realtime_rtsp"),
        importlib.import_module("anytran.runners.run_multi_rtsp"),
    )
    try:
        yield modules
    finally:
        for name, module in original_module_states.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_run_file_input_text_translation_and_audio_outputs(monkeypatch, tmp_path, runner_modules):
    rfi, _, _, _, _ = runner_modules
    input_path = tmp_path / "input.txt"
    input_path.write_text("hola. adios", encoding="utf-8")

    translate_calls = []
    tts_calls = []
    output_calls = []

    def fake_translate_text(text, source_lang=None, target_lang=None, backend=None, verbose=False):
        translate_calls.append((text, source_lang, target_lang, backend, verbose))
        return f"{target_lang}:{text}"

    def fake_split(text):
        return [p.strip() for p in text.split(".") if p.strip()]

    monkeypatch.setattr(rfi, "translate_text", fake_translate_text)
    monkeypatch.setattr(rfi, "split_into_sentences", fake_split)
    monkeypatch.setattr(rfi, "normalize_text", lambda text: text.upper())
    monkeypatch.setattr(rfi, "get_translategemma_model", lambda: None)

    def fake_tts(text, rate, lang, **kwargs):
        tts_calls.append((text, rate, lang))
        return np.array([0.1, 0.2], dtype=np.float32)

    def fake_output_audio(data, path, play=False):
        output_calls.append((path, np.array(data)))

    monkeypatch.setattr(rfi, "synthesize_tts_pcm", fake_tts)
    monkeypatch.setattr(rfi, "output_audio", fake_output_audio)

    scribe_text_file = tmp_path / "scribe.txt"
    slate_text_file = tmp_path / "slate.txt"
    scribe_audio_path = tmp_path / "scribe.wav"
    slate_audio_path = tmp_path / "slate.wav"

    rfi.run_file_input(
        str(input_path),
        input_lang="es",
        text_translation_target="fr",
        slate_backend="translategemma",
        scribe_text_file=str(scribe_text_file),
        slate_text_file=str(slate_text_file),
        output_audio_path=str(scribe_audio_path),
        slate_audio_path=str(slate_audio_path),
        timers_all=True,
        verbose=True,
    )

    assert len(translate_calls) == 3  # two sentences to English, one to French
    assert tts_calls == [
        ("en:hola en:adios", 16000, "en"),
        ("fr:en:hola en:adios", 16000, "fr"),
    ]
    assert all(call[-1] is True for call in translate_calls)
    assert output_calls[0][0] == str(scribe_audio_path)
    assert output_calls[1][0] == str(slate_audio_path)
    np.testing.assert_allclose(output_calls[0][1], np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(output_calls[1][1], np.array([0.1, 0.2], dtype=np.float32))
    assert scribe_text_file.read_text(encoding="utf-8") == "EN:HOLA EN:ADIOS\n"
    assert slate_text_file.read_text(encoding="utf-8") == "FR:EN:HOLA EN:ADIOS\n"


def test_run_file_input_audio_chunk_processing(monkeypatch, tmp_path, runner_modules):
    rfi, _, _, _, _ = runner_modules
    audio_array = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    monkeypatch.setattr(rfi, "load_audio_any", lambda path: (audio_array, 16000))
    monkeypatch.setattr(rfi, "compute_window_params", lambda *args, **kwargs: (2, 0))
    monkeypatch.setattr(rfi, "normalize_text", lambda text: text.upper())

    process_calls = []

    slate_outputs = 0

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        process_calls.append(np.array(audio_segment))
        scribe_segments = kwargs["scribe_tts_segments"]
        slate_segments = kwargs["slate_tts_segments"]
        if scribe_segments is not None:
            scribe_segments.append(np.array([1.0, 1.0], dtype=np.float32))
        if slate_segments is not None:
            slate_segments.append(np.array([2.0, 2.0], dtype=np.float32))
        idx = len(process_calls)
        # Return the same slate output so deduplication keeps only the first write.
        nonlocal slate_outputs
        slate_outputs += 1
        return {"scribe": f"scribe-{idx}", "slate": "SLATE-UNCHANGED"}

    output_calls = []
    monkeypatch.setattr(rfi, "process_audio_chunk", fake_process_audio_chunk)
    monkeypatch.setattr(rfi, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    scribe_text_file = tmp_path / "scribe_audio.txt"
    slate_text_file = tmp_path / "slate_audio.txt"
    scribe_audio_path = tmp_path / "scribe_audio.wav"
    slate_audio_path = tmp_path / "slate_audio.wav"

    rfi.run_file_input(
        str(tmp_path / "audio.wav"),
        scribe_text_file=str(scribe_text_file),
        slate_text_file=str(slate_text_file),
        output_audio_path=str(scribe_audio_path),
        slate_audio_path=str(slate_audio_path),
    )

    assert len(process_calls) == 2
    assert slate_outputs == 2  # both chunks produced slate output but only one was written
    assert scribe_text_file.read_text(encoding="utf-8").splitlines() == ["SCRIBE-1", "SCRIBE-2"]
    assert slate_text_file.read_text(encoding="utf-8").splitlines() == ["SLATE-UNCHANGED"]
    assert output_calls[0][0] == str(scribe_audio_path)
    np.testing.assert_allclose(output_calls[0][1], np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert output_calls[1][0] == str(slate_audio_path)
    np.testing.assert_allclose(output_calls[1][1], np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))


def test_run_realtime_output_non_windows_returns(monkeypatch, runner_modules):
    _, rro, _, _, _ = runner_modules
    monkeypatch.setattr(sys, "platform", "linux")
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not be called")
    monkeypatch.setattr(rro, "get_wasapi_loopback_device_info", _fail_if_called)
    assert rro.run_realtime_output() is None


def test_run_realtime_output_processes_chunks(monkeypatch, tmp_path, runner_modules):
    _, rro, _, _, _ = runner_modules
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(rro.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rro, "get_wasapi_loopback_device_info", lambda *args, **kwargs: {"id": "loop"})
    monkeypatch.setattr(rro, "compute_window_params", lambda *args, **kwargs: (2, 0))
    monkeypatch.setattr(rro, "normalize_text", lambda text: text.upper())

    mqtt_calls = []
    monkeypatch.setattr(rro, "init_mqtt", lambda *args, **kwargs: mqtt_calls.append(args))

    output_calls = []
    monkeypatch.setattr(rro, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    process_calls = []

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        process_calls.append(np.array(audio_segment))
        scribe_segments = kwargs["scribe_tts_segments"]
        slate_segments = kwargs["slate_tts_segments"]
        if scribe_segments is not None:
            scribe_segments.append(np.array([1.0, 1.0], dtype=np.float32))
        if slate_segments is not None:
            slate_segments.append(np.array([2.0, 2.0], dtype=np.float32))
        return {"scribe": "LINE", "slate": "LINE"}

    monkeypatch.setattr(rro, "process_audio_chunk", fake_process_audio_chunk)

    def fake_stream_output_audio(queue, device_info, rate, stop_flag, verbose):
        queue.put(np.array([0.1, 0.2], dtype=np.float32))
        queue.put(np.array([0.3, 0.4], dtype=np.float32))
        time.sleep(0.05)
        stop_flag.set()

    monkeypatch.setattr(rro, "stream_output_audio", fake_stream_output_audio)

    scribe_text_file = tmp_path / "rt_output_scribe.txt"
    slate_text_file = tmp_path / "rt_output_slate.txt"
    scribe_audio_path = tmp_path / "rt_output_scribe.wav"
    slate_audio_path = tmp_path / "rt_output_slate.wav"

    rro.run_realtime_output(
        mqtt_broker="localhost",
        scribe_text_file=str(scribe_text_file),
        slate_text_file=str(slate_text_file),
        output_audio_path=str(scribe_audio_path),
        slate_audio_path=str(slate_audio_path),
        dedup=True,
    )

    assert mqtt_calls  # init_mqtt was invoked
    assert scribe_text_file.read_text(encoding="utf-8").splitlines() == ["LINE"]
    assert slate_text_file.read_text(encoding="utf-8").splitlines() == ["LINE"]
    assert len(process_calls) == 2  # two chunks processed while dedup kept a single text write
    assert output_calls[0][0] == str(scribe_audio_path)
    np.testing.assert_allclose(output_calls[0][1], np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert output_calls[1][0] == str(slate_audio_path)
    np.testing.assert_allclose(output_calls[1][1], np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))


def test_run_realtime_youtube_invalid_url(monkeypatch, runner_modules):
    _, _, rry, _, _ = runner_modules
    monkeypatch.setattr(rry, "extract_youtube_video_id", lambda url: None)
    called = {}

    def _fail_if_called(*args, **kwargs):
        called["unexpected"] = True
    monkeypatch.setattr(rry, "validate_youtube_video", _fail_if_called)

    assert rry.run_realtime_youtube("bad-url", "key") is None
    assert "unexpected" not in called


def test_run_realtime_youtube_processes_and_flushes_buffer(monkeypatch, tmp_path, runner_modules):
    _, _, rry, _, _ = runner_modules
    monkeypatch.setattr(rry, "extract_youtube_video_id", lambda url: "vid123")
    validate_calls = {}
    def fake_validate(api_key, vid, verbose=False):
        validate_calls["called"] = verbose
        return {"contentDetails": {"duration": "PT5S"}}
    monkeypatch.setattr(rry, "validate_youtube_video", fake_validate)
    monkeypatch.setattr(rry, "parse_iso8601_duration", lambda duration: 5)
    monkeypatch.setattr(rry.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rry, "compute_window_params", lambda *args, **kwargs: (4, 1))
    monkeypatch.setattr(rry, "normalize_text", lambda text: text.upper())
    monkeypatch.setattr(rry.shutil, "which", lambda name: "runtime" if name == "node" else None)
    monkeypatch.setattr(rry, "get_youtube_audio_stream_url", lambda *args, **kwargs: "http://example.com/audio")

    mqtt_calls = []
    monkeypatch.setattr(rry, "init_mqtt", lambda *args, **kwargs: mqtt_calls.append(args))

    def fake_stream_youtube_audio(resolve_audio_url, audio_queue, rate, stop_flag, expected_duration, max_retries, verbose, _):
        resolve_audio_url()
        audio_queue.put(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        audio_queue.put(np.array([0.5, 0.6, 0.7], dtype=np.float32))
        time.sleep(0.05)
        stop_flag.set()

    monkeypatch.setattr(rry, "stream_youtube_audio", fake_stream_youtube_audio)

    call_count = {"n": 0}

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        call_count["n"] += 1
        scribe_segments = kwargs["scribe_tts_segments"]
        slate_segments = kwargs["slate_tts_segments"]
        if scribe_segments is not None:
            scribe_segments.append(np.array([float(call_count["n"])], dtype=np.float32))
        if slate_segments is not None:
            slate_segments.append(np.array([float(call_count["n"] * 2)], dtype=np.float32))
        return {"scribe": f"scribe-{call_count['n']}", "slate": f"slate-{call_count['n']}"}

    monkeypatch.setattr(rry, "process_audio_chunk", fake_process_audio_chunk)
    output_calls = []
    monkeypatch.setattr(rry, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    scribe_text_file = tmp_path / "yt_scribe.txt"
    slate_text_file = tmp_path / "yt_slate.txt"
    scribe_audio_path = tmp_path / "yt_scribe.wav"
    slate_audio_path = tmp_path / "yt_slate.wav"

    rry.run_realtime_youtube(
        "https://youtube.com/watch?v=vid123",
        "api-key",
        mqtt_broker="localhost",
        scribe_text_file=str(scribe_text_file),
        slate_text_file=str(slate_text_file),
        output_audio_path=str(scribe_audio_path),
        slate_audio_path=str(slate_audio_path),
        verbose=True,
    )

    assert mqtt_calls
    assert call_count["n"] == 3  # two chunks in loop, one final buffer
    assert scribe_text_file.read_text(encoding="utf-8").splitlines() == ["SCRIBE-1", "SCRIBE-2", "SCRIBE-3"]
    assert slate_text_file.read_text(encoding="utf-8").splitlines() == ["SLATE-1", "SLATE-2", "SLATE-3"]
    assert validate_calls["called"] is True
    assert output_calls[0][0] == str(scribe_audio_path)
    np.testing.assert_allclose(output_calls[0][1], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert output_calls[1][0] == str(slate_audio_path)
    np.testing.assert_allclose(output_calls[1][1], np.array([2.0, 4.0, 6.0], dtype=np.float32))


@pytest.mark.parametrize(
    "dedup_flag,expected_lines",
    [
        (False, ["same", "same"]),
        (True, ["same"]),
    ],
)
def test_run_realtime_youtube_respects_dedup_flag(monkeypatch, tmp_path, runner_modules, dedup_flag, expected_lines):
    _, _, rry, _, _ = runner_modules
    monkeypatch.setattr(rry, "extract_youtube_video_id", lambda url: "vid123")
    monkeypatch.setattr(rry, "validate_youtube_video", lambda *args, **kwargs: {"contentDetails": {"duration": "PT5S"}})
    monkeypatch.setattr(rry, "parse_iso8601_duration", lambda duration: 5)
    monkeypatch.setattr(rry.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rry, "compute_window_params", lambda *args, **kwargs: (2, 0))
    monkeypatch.setattr(rry.shutil, "which", lambda name: "runtime" if name == "node" else None)
    monkeypatch.setattr(rry, "get_youtube_audio_stream_url", lambda *args, **kwargs: "http://example.com/audio")

    def fake_stream(resolve_audio_url, audio_queue, rate, stop_flag, expected_duration, max_retries, verbose, _):
        resolve_audio_url()
        audio_queue.put(np.array([0.1, 0.2], dtype=np.float32))
        audio_queue.put(np.array([0.3], dtype=np.float32))
        # Signal that streaming is complete so the runner can exit promptly
        stop_flag.set()
        time.sleep(0.05)
        stop_flag.set()

    monkeypatch.setattr(rry, "stream_youtube_audio", fake_stream)

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        return {"scribe": "same", "slate": "same"}

    monkeypatch.setattr(rry, "process_audio_chunk", fake_process_audio_chunk)

    scribe_text_file = tmp_path / "yt_flag_scribe.txt"
    slate_text_file = tmp_path / "yt_flag_slate.txt"

    rry.run_realtime_youtube(
        "https://youtube.com/watch?v=vid123",
        "api-key",
        scribe_text_file=str(scribe_text_file),
        slate_text_file=str(slate_text_file),
        dedup=dedup_flag,
        normalize=False,
    )

    assert scribe_text_file.read_text(encoding="utf-8").splitlines() == expected_lines
    assert slate_text_file.read_text(encoding="utf-8").splitlines() == expected_lines


# --------------------------------------------------------------------------
# Tests for --capture-voice (capture original input audio to file)
# --------------------------------------------------------------------------

def test_run_realtime_rtsp_capture_voice(monkeypatch, tmp_path, runner_modules):
    """run_realtime_rtsp saves raw input audio when capture_voice_path is set."""
    _, _, _, rrr, _ = runner_modules
    monkeypatch.setattr(rrr.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rrr, "compute_window_params", lambda *args, **kwargs: (2, 0))
    monkeypatch.setattr(rrr, "normalize_text", lambda text: text)
    # Avoid 0.5-second sleep in the KeyboardInterrupt handler
    monkeypatch.setattr(rrr.time, "sleep", lambda _: None)

    output_calls = []
    monkeypatch.setattr(rrr, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    call_count = {"n": 0}

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        call_count["n"] += 1
        # Stop the loop after 2 chunks by raising KeyboardInterrupt
        if call_count["n"] >= 2:
            raise KeyboardInterrupt()
        return {"scribe": "hello", "slate": None}

    monkeypatch.setattr(rrr, "process_audio_chunk", fake_process_audio_chunk)

    def fake_stream_rtsp_audio(rtsp_url, audio_queue):
        audio_queue.put(np.array([0.1, 0.2], dtype=np.float32))
        audio_queue.put(np.array([0.3, 0.4], dtype=np.float32))

    monkeypatch.setattr(rrr, "stream_rtsp_audio", fake_stream_rtsp_audio)

    capture_path = tmp_path / "captured.wav"
    rrr.run_realtime_rtsp(
        "rtsp://example.com/stream",
        capture_voice_path=str(capture_path),
    )

    assert len(output_calls) == 1
    assert output_calls[0][0] == str(capture_path)
    np.testing.assert_allclose(output_calls[0][1], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_run_realtime_output_capture_voice(monkeypatch, tmp_path, runner_modules):
    """run_realtime_output saves raw input audio when capture_voice_path is set."""
    _, rro, _, _, _ = runner_modules
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(rro.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rro, "get_wasapi_loopback_device_info", lambda *args, **kwargs: {"id": "loop"})
    monkeypatch.setattr(rro, "compute_window_params", lambda *args, **kwargs: (2, 0))
    monkeypatch.setattr(rro, "normalize_text", lambda text: text)

    output_calls = []
    monkeypatch.setattr(rro, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        return {"scribe": "hello", "slate": None}

    monkeypatch.setattr(rro, "process_audio_chunk", fake_process_audio_chunk)

    def fake_stream_output_audio(queue, device_info, rate, stop_flag, verbose):
        queue.put(np.array([0.1, 0.2], dtype=np.float32))
        queue.put(np.array([0.3, 0.4], dtype=np.float32))
        time.sleep(0.05)
        stop_flag.set()

    monkeypatch.setattr(rro, "stream_output_audio", fake_stream_output_audio)

    capture_path = tmp_path / "captured_output.wav"
    rro.run_realtime_output(capture_voice_path=str(capture_path))

    assert len(output_calls) == 1
    assert output_calls[0][0] == str(capture_path)
    np.testing.assert_allclose(output_calls[0][1], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_run_realtime_youtube_capture_voice(monkeypatch, tmp_path, runner_modules):
    """run_realtime_youtube saves raw input audio when capture_voice_path is set."""
    _, _, rry, _, _ = runner_modules
    monkeypatch.setattr(rry, "extract_youtube_video_id", lambda url: "vid123")
    monkeypatch.setattr(rry, "validate_youtube_video", lambda *args, **kwargs: {"contentDetails": {"duration": "PT5S"}})
    monkeypatch.setattr(rry, "parse_iso8601_duration", lambda duration: 5)
    monkeypatch.setattr(rry.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rry, "compute_window_params", lambda *args, **kwargs: (2, 0))
    monkeypatch.setattr(rry, "normalize_text", lambda text: text)
    monkeypatch.setattr(rry.shutil, "which", lambda name: None)
    monkeypatch.setattr(rry, "get_youtube_audio_stream_url", lambda *args, **kwargs: "http://example.com/audio")

    def fake_stream_youtube_audio(resolve_audio_url, audio_queue, rate, stop_flag, expected_duration, max_retries, verbose, _):
        resolve_audio_url()
        audio_queue.put(np.array([0.1, 0.2], dtype=np.float32))
        audio_queue.put(np.array([0.3, 0.4], dtype=np.float32))
        time.sleep(0.05)
        stop_flag.set()

    monkeypatch.setattr(rry, "stream_youtube_audio", fake_stream_youtube_audio)

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        return {"scribe": "hello", "slate": None}

    monkeypatch.setattr(rry, "process_audio_chunk", fake_process_audio_chunk)

    output_calls = []
    monkeypatch.setattr(rry, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    capture_path = tmp_path / "captured_yt.wav"
    rry.run_realtime_youtube(
        "https://youtube.com/watch?v=vid123",
        "api-key",
        capture_voice_path=str(capture_path),
    )

    assert len(output_calls) == 1
    assert output_calls[0][0] == str(capture_path)
    np.testing.assert_allclose(output_calls[0][1], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_run_multi_rtsp_capture_voice(monkeypatch, tmp_path, runner_modules):
    """run_multi_rtsp saves per-stream raw input audio when capture_voice_path is set."""
    _, _, _, _, rmr = runner_modules
    monkeypatch.setattr(rmr.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rmr, "compute_window_params", lambda *args, **kwargs: (2, 0))
    monkeypatch.setattr(rmr, "normalize_text", lambda text: text)

    output_calls = []
    monkeypatch.setattr(rmr, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    call_counts = {}

    def fake_process_audio_chunk(audio_segment, rate, *args, stream_id=None, **kwargs):
        key = stream_id or "unknown"
        call_counts[key] = call_counts.get(key, 0) + 1
        if call_counts[key] >= 2:
            raise KeyboardInterrupt()
        return {"scribe": "hello", "slate": None}

    monkeypatch.setattr(rmr, "process_audio_chunk", fake_process_audio_chunk)

    def fake_stream_rtsp_audio(rtsp_url, audio_queue):
        audio_queue.put(np.array([0.1, 0.2], dtype=np.float32))
        audio_queue.put(np.array([0.3, 0.4], dtype=np.float32))

    monkeypatch.setattr(rmr, "stream_rtsp_audio", fake_stream_rtsp_audio)

    capture_path = tmp_path / "captured_multi.wav"
    rmr.run_multi_rtsp(
        ["rtsp://cam1", "rtsp://cam2"],
        capture_voice_path=str(capture_path),
    )

    # Each stream saves a per-stream file like captured_multi_stream1.wav and captured_multi_stream2.wav
    saved_paths = {c[0] for c in output_calls}
    assert str(tmp_path / "captured_multi_stream1.wav") in saved_paths
    assert str(tmp_path / "captured_multi_stream2.wav") in saved_paths
    for path, data in output_calls:
        np.testing.assert_allclose(data, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_capture_voice_not_called_when_path_is_none(monkeypatch, tmp_path, runner_modules):
    """When capture_voice_path is None, output_audio is not called for capture."""
    _, rro, _, _, _ = runner_modules
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(rro.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(rro, "get_wasapi_loopback_device_info", lambda *args, **kwargs: {"id": "loop"})
    monkeypatch.setattr(rro, "compute_window_params", lambda *args, **kwargs: (2, 0))

    output_calls = []
    monkeypatch.setattr(rro, "output_audio", lambda data, path, play=False: output_calls.append((path, np.array(data))))

    def fake_process_audio_chunk(audio_segment, rate, *args, **kwargs):
        return {"scribe": "hello", "slate": None}

    monkeypatch.setattr(rro, "process_audio_chunk", fake_process_audio_chunk)

    def fake_stream_output_audio(queue, device_info, rate, stop_flag, verbose):
        queue.put(np.array([0.1, 0.2], dtype=np.float32))
        time.sleep(0.05)
        stop_flag.set()

    monkeypatch.setattr(rro, "stream_output_audio", fake_stream_output_audio)

    # No capture_voice_path — output_audio should not be called
    rro.run_realtime_output(capture_voice_path=None)

    assert output_calls == []


def test_run_file_input_same_lang_skips_stage2_translation(monkeypatch, tmp_path, runner_modules):
    """When input_lang == text_translation_target, Stage 2 translation is skipped
    and the original input text is written directly to the slate file."""
    rfi, _, _, _, _ = runner_modules
    original_text = "bonjour. au revoir"
    input_path = tmp_path / "input.txt"
    input_path.write_text(original_text, encoding="utf-8")

    translate_calls = []

    def fake_translate_text(text, source_lang=None, target_lang=None, backend=None, verbose=False):
        translate_calls.append((text, source_lang, target_lang))
        return f"{target_lang}:{text}"

    def fake_split(text):
        return [p.strip() for p in text.split(".") if p.strip()]

    monkeypatch.setattr(rfi, "translate_text", fake_translate_text)
    monkeypatch.setattr(rfi, "split_into_sentences", fake_split)
    monkeypatch.setattr(rfi, "normalize_text", lambda t: t)

    scribe_text_file = tmp_path / "scribe.txt"
    slate_text_file = tmp_path / "slate.txt"

    rfi.run_file_input(
        str(input_path),
        input_lang="fr",
        text_translation_target="fr",
        slate_backend="googletrans",
        scribe_text_file=str(scribe_text_file),
        slate_text_file=str(slate_text_file),
        verbose=True,
    )

    # Stage 1 should still run (fr -> en) for the scribe, but Stage 2 should be skipped.
    stage2_calls = [c for c in translate_calls if c[2] == "fr"]
    assert stage2_calls == [], "Stage 2 (en->fr) translation should be skipped when input_lang == target_lang"

    # Slate should contain the original French text, not a round-trip translation.
    slate_content = slate_text_file.read_text(encoding="utf-8").strip()
    assert slate_content == original_text.strip()


def test_run_file_input_same_lang_with_dialect_skips_stage2_translation(monkeypatch, tmp_path, runner_modules):
    """When input_lang and text_translation_target share the same base language code
    (e.g. 'zh-CN' vs 'zh'), Stage 2 translation is still skipped."""
    rfi, _, _, _, _ = runner_modules
    original_text = "你好世界"
    input_path = tmp_path / "input.txt"
    input_path.write_text(original_text, encoding="utf-8")

    translate_calls = []

    def fake_translate_text(text, source_lang=None, target_lang=None, backend=None, verbose=False):
        translate_calls.append((text, source_lang, target_lang))
        return f"{target_lang}:{text}"

    monkeypatch.setattr(rfi, "translate_text", fake_translate_text)
    monkeypatch.setattr(rfi, "split_into_sentences", lambda t: [t])
    monkeypatch.setattr(rfi, "normalize_text", lambda t: t)

    slate_text_file = tmp_path / "slate_zh.txt"

    rfi.run_file_input(
        str(input_path),
        input_lang="zh-CN",
        text_translation_target="zh",
        slate_backend="googletrans",
        slate_text_file=str(slate_text_file),
    )

    stage2_calls = [c for c in translate_calls if c[1] == "en"]
    assert stage2_calls == [], "Stage 2 translation should be skipped when base language codes match"

    slate_content = slate_text_file.read_text(encoding="utf-8").strip()
    assert slate_content == original_text.strip()


def test_run_file_input_slate_no_opt_forces_stage2_translation(monkeypatch, tmp_path, runner_modules):
    """When slate_no_opt=True, Stage 2 translation runs even if input_lang == text_translation_target."""
    rfi, _, _, _, _ = runner_modules
    original_text = "bonjour. au revoir"
    input_path = tmp_path / "input.txt"
    input_path.write_text(original_text, encoding="utf-8")

    translate_calls = []

    def fake_translate_text(text, source_lang=None, target_lang=None, backend=None, verbose=False):
        translate_calls.append((text, source_lang, target_lang))
        return f"{target_lang}:{text}"

    def fake_split(text):
        return [p.strip() for p in text.split(".") if p.strip()]

    monkeypatch.setattr(rfi, "translate_text", fake_translate_text)
    monkeypatch.setattr(rfi, "split_into_sentences", fake_split)
    monkeypatch.setattr(rfi, "normalize_text", lambda t: t)

    slate_text_file = tmp_path / "slate.txt"

    rfi.run_file_input(
        str(input_path),
        input_lang="fr",
        text_translation_target="fr",
        slate_backend="googletrans",
        slate_text_file=str(slate_text_file),
        slate_no_opt=True,
    )

    # With slate_no_opt=True the round-trip must happen: fr→en (Stage 1) then en→fr (Stage 2).
    stage2_calls = [c for c in translate_calls if c[2] == "fr"]
    assert len(stage2_calls) > 0, "Stage 2 (en->fr) translation should run when slate_no_opt=True"

    # Slate output should be the Stage-2 translated text, not the raw original.
    slate_content = slate_text_file.read_text(encoding="utf-8").strip()
    assert slate_content != original_text.strip(), "Slate content should differ from original when opt is disabled"
