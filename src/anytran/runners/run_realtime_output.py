"""Realtime-output runner (Windows WASAPI loopback capture).

Captures system audio on Windows and streams it through the translation pipeline.
"""

import signal
import sys
import threading
import time

import numpy as np

from ..pipeline_config import RunnerConfig, StreamContext
from ..utils import compute_window_params
from ..mqtt_client import init_mqtt
from ..normalizer import normalize_text
from ..audio_io import output_audio
from ..stream_output import get_wasapi_loopback_device_info
from ..processing import process_audio_chunk


def run_realtime_output(
    cfg: "RunnerConfig" = None,
    **kwargs,
):
    """Run the pipeline on Windows WASAPI loopback audio.

    Parameters
    ----------
    cfg : RunnerConfig
        Combined runner configuration.  Runner-specific extras:
        * ``output_device`` : str or None  - WASAPI device id override.
        If not provided, individual keyword arguments are accepted for
        backward compatibility.
    **kwargs : dict
        Legacy individual keyword arguments.
    """
    if cfg is None:
        cfg = RunnerConfig._from_kwargs(**kwargs)
    pipeline = cfg.pipeline
    output = cfg.output
    mqtt = cfg.mqtt

    # -- Windows-only -------------------------
    if sys.platform != "win32":
        print("Realtime output is only supported on Windows.")
        return

    verbose = pipeline.verbose
    normalize = pipeline.normalize
    dedup = pipeline.dedup
    scribe_vad = pipeline.scribe_vad
    magnitude_threshold = pipeline.magnitude_threshold

    # -- output paths -------------------------
    scribe_text_file = output.scribe_text_file
    slate_text_file = output.slate_text_file
    output_audio_path = output.output_audio_path
    slate_audio_path = output.slate_audio_path
    capture_voice_path = output.capture_voice_path

    # -- mqtt -------------------------
    topic = mqtt.topic if mqtt.is_enabled else None
    _ = mqtt

    # -- stream context -----------------------
    stream_ctx = StreamContext()

    # -- get device -------------------------
    device_info = _get_wasapi_loopback_device_info(cfg.get("output_device"))
    if device_info is None:
        print("Could not find a WASAPI loopback device.")
        return

    sample_rate = device_info.get("sample_rate", 48000)
    channels = device_info.get("channels", 1)

    # -- compute window/step -------------------------
    window_size, hop_size = compute_window_params(
        sample_rate,
        window_seconds=pipeline.window_seconds,
        overlap_seconds=pipeline.overlap_seconds,
    )

    # -- silence-detection parameters -------------------------
    mag_threshold = magnitude_threshold

    last_scribe_text = None
    last_slate_text = None

    # -- capture-voice accumulator -------------------------
    capture_chunks = [] if capture_voice_path else None

    # -- mqtt init -------------------------
    if mqtt.is_enabled:
        mqtt_client = init_mqtt(
            broker=mqtt.broker,
            port=mqtt.port,
            username=mqtt.username,
            password=mqtt.password,
            topic=topic,
        )
    else:
        mqtt_client = None

    # -- audio queue -------------------------
    audio_queue = _start_wasapi_capture(device_info, sample_rate, channels)

    stop_flag = threading.Event()

    try:
        while not stop_flag.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
            except Exception:
                continue

            # -- down-mix to mono -------------------------
            if channels > 1:
                chunk = np.mean(chunk.reshape(-1, channels), axis=1)

            # -- silence gate -------------------------
            if not scribe_vad:
                max_abs = np.max(np.abs(chunk))
                if max_abs < mag_threshold:
                    continue

            result = _process_chunk(
                chunk,
                sample_rate,
                pipeline,
                stream_ctx,
                mqtt_client,
            )

            scribe_text = result.get("scribe")
            slate_text = result.get("slate")

            # -- dedup -------------------------
            if dedup:
                if scribe_text == last_scribe_text:
                    scribe_text = None
                if slate_text == last_slate_text:
                    slate_text = None
                last_scribe_text = result.get("scribe")
                last_slate_text = result.get("slate")

            # -- write text to files -------------------------
            if scribe_text and scribe_text_file:
                with open(scribe_text_file, "a", encoding="utf-8") as f:
                    f.write(scribe_text + "\n")
            if slate_text and slate_text_file:
                with open(slate_text_file, "a", encoding="utf-8") as f:
                    f.write(slate_text + "\n")

            # -- accumulate capture-voice chunks -------------------------
            if capture_chunks is not None:
                capture_chunks.append(chunk)

    except KeyboardInterrupt:
        stop_flag.set()
    finally:
        # -- write capture-voice file -------------------------
        if capture_chunks:
            capture_audio = np.concatenate(capture_chunks, axis=0)
            output_audio(capture_audio, capture_voice_path)

        # -- write accumulated TTS audio -------------------------
        if stream_ctx.scribe_tts_segments and output_audio_path:
            _write_accumulated_audio(stream_ctx.scribe_tts_segments, output_audio_path)
        if stream_ctx.slate_tts_segments and slate_audio_path:
            _write_accumulated_audio(stream_ctx.slate_tts_segments, slate_audio_path)


# ------ Helpers ------

def _process_chunk(chunk, sample_rate, pipeline, stream_ctx, mqtt_cfg):
    return process_audio_chunk(
        chunk,
        sample_rate,
        pipeline,
        stream_ctx,
        mqtt_cfg,
    )


def _get_wasapi_loopback_device_info(device_id=None):
    """Return loopback device info dict or None."""
    try:
        import sounddevice as sd
    except ImportError:
        return None
    devices = sd.query_devices()
    for dev in devices:
        if device_id and dev["name"] == device_id:
            return dev
        if "loopback" in dev.get("name", "").lower():
            return dev
    return None


def _start_wasapi_capture(device_info, sample_rate, channels):
    """Start a background WASAPI capture thread, return an queue."""
    import queue
    audio_queue = queue.Queue()

    def capture_worker():
        import sounddevice as sd
        with sd.InputStream(
            device=device_info.get("name"),
            channels=channels,
            samplerate=sample_rate,
            dtype="float32",
        ) as stream:
            while not threading.current_thread().daemon:
                data, _ = stream.read(int(stream.samplerate * 0.5))
                audio_queue.put(data.copy())

    t = threading.Thread(target=capture_worker, daemon=True)
    t.start()
    return audio_queue


def _write_accumulated_audio(segments, path):
    if not segments:
        return
    audio_data = np.concatenate(segments, axis=0)
    output_audio(audio_data, path)