"""Realtime RTSP runner.

Connects to a single RTSP stream and runs the translation pipeline on it.
"""

import threading
import signal
import time

import numpy as np

from ..pipeline_config import RunnerConfig, StreamContext
from ..chatlog import ChatLogger, extract_ip_from_rtsp_url
from ..mqtt_client import init_mqtt
from ..normalizer import normalize_text
from ..processing import process_audio_chunk
from ..stream_rtsp import stream_rtsp_audio
from ..timing import TimingsAggregator
from ..audio_io import output_audio
from ..utils import compute_window_params
from queue import Queue, Empty


def run_realtime_rtsp(
    rtsp_url: str,
    cfg: "RunnerConfig" = None,
    **kwargs,
):
    """Run the pipeline on a single RTSP stream.

    Parameters
    ----------
    rtsp_url : str
        The RTSP stream URL.
    cfg : RunnerConfig
        Combined runner configuration.
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

    verbose = pipeline.verbose
    normalize = pipeline.normalize
    dedup = pipeline.dedup
    timers_all = pipeline.timers_all

    # -- output paths ------
    scribe_text_file = output.scribe_text_file
    slate_text_file = output.slate_text_file
    output_audio_path = output.output_audio_path
    slate_audio_path = output.slate_audio_path
    capture_voice_path = output.capture_voice_path
    chat_log_dir = output.chat_log_dir

    print("Starting real-time RTSP audio translation.")
    print(f"Input language: {pipeline.input_lang}, Output language: {pipeline.output_lang}")
    if output_audio_path:
        print(f"Output audio will be saved to: {output_audio_path}")
    if scribe_text_file:
        print(f"Stage 1 (English) text will be saved to: {scribe_text_file}")
    if slate_text_file:
        print(f"Stage 2 (translated) text will be saved to: {slate_text_file}")
    if capture_voice_path:
        print(f"Original input voice will be saved to: {capture_voice_path}")
    if mqtt.is_enabled:
        print(f"MQTT output enabled: {mqtt.broker}:{mqtt.port}, topic: {mqtt.topic}")
    print("Press Ctrl+C to stop.")

    stop_flag = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping...", flush=True)
        stop_flag.set()

    signal.signal(signal.SIGINT, signal_handler)

    # -- chat logger ------
    chat_logger = None
    rtsp_ip = None
    if chat_log_dir:
        chat_logger = ChatLogger(chat_log_dir)
        rtsp_ip = extract_ip_from_rtsp_url(rtsp_url)
        print(f"Chat logging enabled. Logs will be saved to: {chat_log_dir}")
        print(f"RTSP IP: {rtsp_ip}")

    # -- mqtt init ------
    if mqtt.is_enabled:
        init_mqtt(mqtt.broker, mqtt.port, mqtt.username, mqtt.password, mqtt.topic)

    # -- timing ------
    pipeline.timers = pipeline.timers or timers_all
    timing_stats = TimingsAggregator("rtsp") if pipeline.timers else None

    # -- mqtt config for downstream calls ------
    mqtt_cfg = mqtt if mqtt.is_enabled else None

    # -- audio queue ------
    audio_queue = Queue(maxsize=50)
    buffer = np.array([], dtype=np.float32)

    # -- file handles ------
    scribe_file = open(scribe_text_file, mode="w", encoding="utf-8") if scribe_text_file else None
    slate_file = open(slate_text_file, mode="w", encoding="utf-8") if slate_text_file else None

    # -- audio segment accumulators ------
    scribe_audio_segments = [] if output_audio_path else None
    slate_audio_segments = [] if slate_audio_path else None
    capture_voice_segments = [] if capture_voice_path else None

    # -- deduplication tracking ------
    recent_slate_outputs = []
    recent_scribe_outputs = []
    dedup_window_size = 10

    # -- start RTSP stream ------
    stream_thread = threading.Thread(
        target=stream_rtsp_audio,
        args=(rtsp_url, audio_queue),
        daemon=True,
    )
    stream_thread.start()

    rate = 16000
    chunk, overlap = compute_window_params(rate, pipeline.window_seconds, pipeline.overlap_seconds)

    try:
        while not stop_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=1)
            except Empty:
                if stop_flag.is_set():
                    break
                continue
            except Exception as exc:
                if stop_flag.is_set():
                    break
                print(f"Error during realtime RTSP processing loop: {exc}", flush=True)
                continue

            if capture_voice_segments is not None:
                capture_voice_segments.append(audio_chunk.copy())

            buffer = np.concatenate([buffer, audio_chunk])

            if len(buffer) >= chunk:
                audio_segment = buffer[:chunk]
                buffer = buffer[chunk - overlap:]

                # -- build per-stream context (mutable state only) ------
                stream_ctx = StreamContext(
                    stream_id="rtsp",
                    chat_logger=chat_logger,
                    rtsp_ip=rtsp_ip,
                    timing_stats=timing_stats,
                    scribe_tts_segments=scribe_audio_segments,
                    slate_tts_segments=slate_audio_segments,
                )

                result = process_audio_chunk(
                    audio_segment,
                    rate,
                    pipeline,
                    stream_ctx,
                    mqtt_cfg,
                )

                # -- write text outputs ------
                if result:
                    scribe_output = result.get("scribe")
                    slate_output = result.get("slate")
                    _write_outputs(
                        scribe_output, slate_output,
                        scribe_file, slate_file,
                        normalize, dedup,
                        recent_scribe_outputs, recent_slate_outputs,
                        dedup_window_size,
                    )

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        stop_flag.set()
        time.sleep(0.5)
    finally:
        if chat_logger:
            chat_logger.close()
        if scribe_file:
            scribe_file.close()
        if slate_file:
            slate_file.close()

        # -- write audio files ------
        _write_audio_segments(scribe_audio_segments, output_audio_path, "Scribe")
        _write_audio_segments(slate_audio_segments, slate_audio_path, "Slate")
        _write_audio_segments(capture_voice_segments, capture_voice_path, "Captured input voice")

        # -- timing summary ------
        if timing_stats is not None:
            stage_summary = timing_stats.format_stage_summary()
            if stage_summary:
                print(f"\nTiming summary (rtsp):\n{stage_summary}")


# -------  Helpers  -------

def _write_outputs(
    scribe_output, slate_output,
    scribe_file, slate_file,
    normalize, dedup,
    recent_scribe, recent_slate,
    dedup_window_size,
):
    """Write scribe/slate text to files, optionally deduplicating."""
    if dedup:
        if scribe_output and scribe_output not in recent_scribe:
            if scribe_file:
                if normalize:
                    scribe_output = normalize_text(scribe_output)
                scribe_file.write(f"{scribe_output}\n")
                scribe_file.flush()
            recent_scribe.append(scribe_output)
            if len(recent_scribe) > dedup_window_size:
                recent_scribe.pop(0)
        if slate_output and slate_output not in recent_slate:
            if slate_file:
                if normalize:
                    slate_output = normalize_text(slate_output)
                slate_file.write(f"{slate_output}\n")
                slate_file.flush()
            recent_slate.append(slate_output)
            if len(recent_slate) > dedup_window_size:
                recent_slate.pop(0)
    else:
        if scribe_output and scribe_file:
            if normalize:
                scribe_output = normalize_text(scribe_output)
            scribe_file.write(f"{scribe_output}\n")
            scribe_file.flush()
        if slate_output and slate_file:
            if normalize:
                slate_output = normalize_text(slate_output)
            slate_file.write(f"{slate_output}\n")
            slate_file.flush()


def _write_audio_segments(segments, path, label):
    """Concatenate and write accumulated audio segments to a file."""
    if segments is not None and len(segments) > 0:
        all_audio = np.concatenate(segments)
        try:
            output_audio(all_audio, path, play=False)
            print(f"{label} audio file saved: {path}", flush=True)
        except Exception as exc:
            print(f"Error saving {label.lower()} audio file: {exc}", flush=True)
            import traceback
            traceback.print_exc()