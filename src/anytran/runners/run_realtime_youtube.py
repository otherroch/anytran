"""Realtime-Youtube runner.

Extracts audio from a YouTube live stream (via yt-dlp), runs the translation
pipeline, and writes output audio/text to disk or publishes via MQTT.
"""

import subprocess
import threading
import time

import numpy as np

from ..pipeline_config import MQTTConfig, PipelineConfig, OutputConfig, RunnerConfig, StreamContext
from ..stream_output import (
    compute_window_params,
    init_mqtt,
    normalize_text,
    output_audio,
)


def run_realtime_youtube(
    url: str,
    cfg: RunnerConfig,
):
    """Run the pipeline on a YouTube live stream.

    Parameters
    ----------
    url : str
        YouTube URL to extract audio from.
    cfg : RunnerConfig
        Combined runner configuration.  Runner-specific extras:
        * ``youtube_js_runtime`` : str or None  - js_path for yt-dlp.
    """
    pipeline = cfg.pipeline
    output = cfg.output
    mqtt = cfg.mqtt

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

    # -- stream context -----------------------
    stream_ctx = StreamContext()

    # -- mqtt -------------------------
    if mqtt.is_enabled:
        mqtt_client = init_mqtt(
            broker=mqtt.broker,
            port=mqtt.port,
            username=mqtt.username,
            password=mqtt.password,
            topic=mqtt.topic,
        )
        mqtt_cfg = mqtt
    else:
        mqtt_client = None
        mqtt_cfg = None

    # -- start yt-dlp -------------------------
    js_path = cfg.get("youtube_js_runtime")
    proc = _start_yt_dlp(url, js_path)
    if proc is None:
        print("Failed to start yt-dlp.")
        return

    # -- detect sample rate from yt-dlp output -------------------------
    sample_rate = 44100  # yt-dlp default

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

    stop_flag = threading.Event()

    try:
        while not stop_flag.is_set():
            # Read 5 seconds of audio (approx.) from yt-dlp stdout
            chunk_bytes = proc.stdout.read(hop_size * 2)  # float32 = 4 bytes
            if not chunk_bytes:
                break

            chunk = np.frombuffer(chunk_bytes, dtype=np.float32)

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
                mqtt_cfg,
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
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

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
    from ..stream_output import process_audio_chunk

    return process_audio_chunk(
        chunk,
        sample_rate,
        pipeline_cfg=pipeline,
        stream_ctx=stream_ctx,
        mqtt_cfg=mqtt_cfg,
    )


def _start_yt_dlp(url, js_path=None):
    """Start yt-dlp to extract audio as raw PCM (float32)."""
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-bitrate", "none",
        "-o", "-",
        url,
    ]
    if js_path:
        cmd.insert(1, f"--js={js_path}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        print("yt-dlp not found. Install it with: pip install yt-dlp")
        return None


def _write_accumulated_audio(segments, path):
    if not segments:
        return
    audio_data = np.concatenate(segments, axis=0)
    output_audio(audio_data, path)