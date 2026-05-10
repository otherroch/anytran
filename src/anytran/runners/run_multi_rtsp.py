"""Multi-RTSP runner.

Connects to multiple RTSP streams concurrently and runs the translation
pipeline on each one.
"""

import threading
import signal
import time

import numpy as np

from ..pipeline_config import RunnerConfig, StreamContext
from ..processing import process_audio_chunk
from ..stream_rtsp import stream_rtsp_audio
from ..timing import TimingsAggregator
from ..audio_io import output_audio
from ..chatlog import ChatLogger, extract_ip_from_rtsp_url
from ..mqtt_client import init_mqtt
from ..normalizer import normalize_text
from ..utils import compute_window_params
from queue import Queue, Empty


def run_multi_rtsp(
    rtsp_urls: list[str],
    cfg: "RunnerConfig" = None,
    **kwargs,
):
    """Run the pipeline on multiple RTSP streams concurrently.

    Parameters
    ----------
    rtsp_urls : list[str]
        List of RTSP stream URLs.
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

    print(f"Starting {len(rtsp_urls)} RTSP streams...")

    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping all streams...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # -- chat logger ------
    chat_logger = None
    chat_log_dir = output.chat_log_dir
    if chat_log_dir:
        chat_logger = ChatLogger(chat_log_dir)
        print(f"Chat logging enabled. Logs will be saved to: {chat_log_dir}")

    # -- mqtt announcement ------
    if mqtt.is_enabled:
        print(f"MQTT output enabled: {mqtt.broker}:{mqtt.port}")
        for i in range(len(rtsp_urls)):
            print(f"  Stream {i + 1} -> topic: stream{i + 1}")
        init_mqtt(mqtt.broker, mqtt.port, mqtt.username, mqtt.password, mqtt.topic)

    # -- text file announcement ------
    if output.scribe_text_file:
        print(f"Scribe text (English) will be saved to: {output.scribe_text_file}")
    if output.slate_text_file:
        print(f"Slate text (translated) will be saved to: {output.slate_text_file}")

    # -- file handles ------
    scribe_file = open(output.scribe_text_file, mode="w", encoding="utf-8") if output.scribe_text_file else None
    slate_file = open(output.slate_text_file, mode="w", encoding="utf-8") if output.slate_text_file else None

    # -- timing ------
    pipeline.timers = pipeline.timers or timers_all
    timing_stats = TimingsAggregator("multi_rtsp") if pipeline.timers else None

    # -- mqtt config for downstream calls ------
    mqtt_cfg = mqtt if mqtt.is_enabled else None

    # -- worker function ------
    def worker(rtsp_url, idx):
        recent_slate_outputs = []
        recent_scribe_outputs = []
        dedup_window_size = 10
        audio_queue = Queue(maxsize=5)
        buffer = np.array([], dtype=np.float32)
        stream_thread = threading.Thread(
            target=stream_rtsp_audio,
            args=(rtsp_url, audio_queue),
            daemon=True,
        )
        stream_thread.start()

        rate = 16000
        chunk, overlap = compute_window_params(rate, pipeline.window_seconds, pipeline.overlap_seconds)

        local_scribe_audio_segments = [] if output.output_audio_path else None
        local_slate_audio_segments = [] if output.slate_audio_path else None
        local_capture_voice_segments = [] if output.capture_voice_path else None

        rtsp_ip = extract_ip_from_rtsp_url(rtsp_url) if chat_logger else None
        if chat_logger and rtsp_ip:
            print(f"[Stream {idx}] RTSP IP: {rtsp_ip}")

        stream_mqtt_topic = f"stream{idx}"

        try:
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=1)
                    if local_capture_voice_segments is not None:
                        local_capture_voice_segments.append(audio_chunk.copy())
                    buffer = np.concatenate([buffer, audio_chunk])
                    if len(buffer) >= chunk:
                        audio_segment = buffer[:chunk]
                        buffer = buffer[chunk - overlap:]

                        # -- build per-stream context (mutable state only) ------
                        stream_ctx = StreamContext(
                            stream_id=idx,
                            chat_logger=chat_logger,
                            rtsp_ip=rtsp_ip,
                            timing_stats=timing_stats,
                            scribe_tts_segments=local_scribe_audio_segments,
                            slate_tts_segments=local_slate_audio_segments,
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
                except Empty:
                    if stop_event.is_set():
                        break
                    continue
        except KeyboardInterrupt:
            if verbose:
                print(f"[Stream {idx}] Stopped.")
        finally:
            # -- write audio files for this stream ------
            _write_stream_audio(local_scribe_audio_segments,
                               output.output_audio_path, idx, "Scribe")
            _write_stream_audio(local_slate_audio_segments,
                               output.slate_audio_path, idx, "Slate")
            _write_stream_audio(local_capture_voice_segments,
                               output.capture_voice_path, idx, "Captured input voice")

    # -- spawn threads ------
    threads = []
    for i, url in enumerate(rtsp_urls):
        thread = threading.Thread(target=worker, args=(url, i + 1), daemon=False)
        threads.append(thread)
        thread.start()

    try:
        for thread in threads:
            while thread.is_alive() and not stop_event.is_set():
                thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Stopping all streams...", flush=True)
        stop_event.set()
        for idx, thread in enumerate(threads, 1):
            thread.join(timeout=5)
    finally:
        if chat_logger:
            chat_logger.close()
        if scribe_file:
            scribe_file.close()
            print(f"Scribe text file saved: {output.scribe_text_file}", flush=True)
        if slate_file:
            slate_file.close()
            print(f"Slate text file saved: {output.slate_text_file}", flush=True)

        # -- timing summary ------
        if timing_stats is not None:
            if timers_all:
                summary = timing_stats.format_summary()
                if summary:
                    print(f"\nTiming summary (multi-rtsp):\n{summary}")
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (multi-rtsp):\n{stage_summary}")
                overhead = timing_stats.format_translate_overhead("chunk")
                if overhead:
                    print(f"\nTiming translate overhead (multi-rtsp):\n{overhead}")
            elif pipeline.timers:
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (multi-rtsp):\n{stage_summary}")


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


def _write_stream_audio(segments, base_path, idx, label):
    """Write accumulated audio segments for one stream to a file."""
    if segments is not None and len(segments) > 0:
        all_audio = np.concatenate(segments)
        if base_path:
            parts = base_path.rsplit(".", 1)
            output_file = f"{parts[0]}_stream{idx}.{parts[1]}" if len(parts) == 2 else f"{base_path}_stream{idx}"
            try:
                output_audio(all_audio, output_file, play=False)
                print(f"[Stream {idx}] {label} audio saved to {output_file}", flush=True)
            except Exception as exc:
                print(f"[Stream {idx}] Error saving {label.lower()} audio file: {exc}", flush=True)
                import traceback
                traceback.print_exc()