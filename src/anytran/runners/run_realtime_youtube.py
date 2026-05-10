from anytran.stream_youtube import extract_youtube_video_id, validate_youtube_video, parse_iso8601_duration, get_youtube_audio_stream_url, stream_youtube_audio
from anytran.processing import process_audio_chunk
from anytran.mqtt_client import init_mqtt
from anytran.normalizer import normalize_text
from anytran.timing import TimingsAggregator
from anytran.utils import compute_window_params
import threading
import numpy as np
import signal
import shutil
import time
from queue import Queue, Empty

from anytran.audio_io import output_audio
from .config import RunnerConfig

DRAIN_COMPLETE_MSG = "Stop requested; audio queue drain complete."
QUEUE_TIMEOUT_DRAIN = 0
QUEUE_TIMEOUT_NORMAL = 1

def run_realtime_youtube(config):
    # Create a RunnerConfig instance from parameters if needed
    if not isinstance(config, RunnerConfig):
        # If config is not a RunnerConfig, create one from the parameters
        config = RunnerConfig(**config)
    
    print("Starting YouTube audio translation...")
    print(f"Input language: {config.input_lang}, Output language: {config.output_lang}")
    if config.output_audio_path:
        print(f"Output audio will be saved to: {config.output_audio_path}")
    # output_text_file removed
    if config.scribe_text_file:
        print(f"Stage 1 (English) text will be saved to: {config.scribe_text_file}")
    if config.slate_text_file:
        print(f"Stage 2 (translated) text will be saved to: {config.slate_text_file}")
    if config.capture_voice_path:
        print(f"Original input voice will be saved to: {config.capture_voice_path}")
    if config.mqtt_broker:
        print(f"MQTT output enabled: {config.mqtt_broker}:{config.mqtt_port}, topic: {config.mqtt_topic}")
    print("Press Ctrl+C to stop.")
    
    youtube_url = config.youtube_url
    youtube_api_key = config.youtube_api_key

    video_id = extract_youtube_video_id(youtube_url)
    if not video_id:
        print("Error: Unable to parse YouTube video ID from URL.")
        return

    validation = validate_youtube_video(youtube_api_key, video_id, verbose=config.verbose)
    if not validation:
        print("Error: YouTube API validation failed. Check API key and video ID.")
        return

    expected_duration = None
    content_details = validation.get("contentDetails") if isinstance(validation, dict) else None
    if content_details:
        expected_duration = parse_iso8601_duration(content_details.get("duration"))

    js_runtime = config.youtube_js_runtime
    if not js_runtime:
        if shutil.which("node"):
            js_runtime = "node"
        elif shutil.which("deno"):
            js_runtime = "deno"
    if config.verbose and js_runtime:
        print(f"Using yt-dlp JS runtime: {js_runtime}")

    if config.verbose and config.youtube_remote_components:
        print(f"Using yt-dlp remote components: {config.youtube_remote_components}")

    def resolve_audio_url():
        return get_youtube_audio_stream_url(
            youtube_url,
            verbose=config.verbose,
            js_runtime=js_runtime,
            remote_components=config.youtube_remote_components,
        )

    stop_flag = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping...", flush=True)
        stop_flag.set()

    signal.signal(signal.SIGINT, signal_handler)

    if config.mqtt_broker:
        init_mqtt(config.mqtt_broker, config.mqtt_port, config.mqtt_username, config.mqtt_password, config.mqtt_topic)
    
    if config.timers_all:
        config.timers = True  # timers_all implies timers
        
    timing_stats = TimingsAggregator("youtube") if config.timers else None

    audio_queue = Queue(maxsize=5)
    buffer = np.array([], dtype=np.float32)
    # output_text_file removed
    scribe_file = open(config.scribe_text_file, mode="w", encoding="utf-8") if config.scribe_text_file else None
    slate_file = open(config.slate_text_file, mode="w", encoding="utf-8") if config.slate_text_file else None
    scribe_audio_segments = [] if config.output_audio_path else None
    slate_audio_segments = [] if config.slate_audio_path else None
    capture_voice_segments = [] if config.capture_voice_path else None

    # Deduplication tracking: keep a sliding window of recent outputs to catch duplicates
    # across overlapping chunks.
    recent_slate_outputs = [] if config.dedup else None
    recent_scribe_outputs = [] if config.dedup else None
    dedup_window_size = 10  # Check last 10 outputs
    # Track last outputs to avoid duplicating the final buffer writes (see final buffer handling below).
    last_written_scribe = None
    last_written_slate = None
    drain_logged = False

    def log_drain_complete():
        nonlocal drain_logged
        if not drain_logged:
            print(DRAIN_COMPLETE_MSG)
            drain_logged = True

    stream_thread = threading.Thread(
        target=stream_youtube_audio,
        args=(resolve_audio_url, audio_queue, 16000, stop_flag, expected_duration, 5, config.verbose, False),
        daemon=True,
    )
    stream_thread.start()

    rate = 16000
    chunk, overlap = compute_window_params(config.window_seconds, config.overlap_seconds, rate)
    idle_seconds = 0
    max_idle_seconds = 60
    stream_ended = False

    try:
        # Drain audio after stop is requested until the queue is empty (handled via Empty), or until idle timeout after the stream ends.
        while True:
            try:
                queue_timeout = QUEUE_TIMEOUT_DRAIN if stop_flag.is_set() else QUEUE_TIMEOUT_NORMAL
                audio_chunk = audio_queue.get(timeout=queue_timeout)
                idle_seconds = 0
                if capture_voice_segments is not None:
                    capture_voice_segments.append(audio_chunk.copy())
                buffer = np.concatenate([buffer, audio_chunk])
                if len(buffer) >= chunk:
                    audio_segment = buffer[:chunk]
                    buffer = buffer[chunk - overlap :]
                    result = process_audio_chunk(
                        audio_segment,
                        rate,
                        config.input_lang,
                        config.output_lang,
                        config.magnitude_threshold,
                        config.model,
                        config.verbose,
                        config.mqtt_broker,
                        config.mqtt_port,
                        config.mqtt_username,
                        config.mqtt_password,
                        config.mqtt_topic,
                        stream_id="youtube",
                        scribe_vad=config.scribe_vad,
                        voice_backend=config.voice_backend,
                        voice_model=config.voice_model,
                        timers=config.timers,
                        timing_stats=timing_stats,
                        scribe_backend=config.scribe_backend,
                        text_translation_target=config.text_translation_target,
                        slate_backend=config.slate_backend,
                        voice_lang=config.voice_lang,
                        scribe_text_file=None,
                        slate_text_file=None,
                        scribe_tts_segments=scribe_audio_segments,
                        slate_tts_segments=slate_audio_segments,
                        voice_match=config.voice_match,
                        lang_prefix=config.lang_prefix,
                    )
                    
                    # Deduplication: Write outputs only if not in recent window
                    if result:
                        scribe_output = result.get('scribe')
                        slate_output = result.get('slate')

                        if config.dedup:
                            if scribe_output:
                                scribe_key = normalize_text(scribe_output) if config.normalize else scribe_output
                                if scribe_key not in recent_scribe_outputs:
                                    if scribe_file:
                                        scribe_file.write(f"{scribe_key}\n")
                                        scribe_file.flush()
                                    recent_scribe_outputs.append(scribe_key)
                                    if len(recent_scribe_outputs) > dedup_window_size:
                                        recent_scribe_outputs.pop(0)
                                    last_written_scribe = scribe_key

                            if slate_output:
                                slate_key = normalize_text(slate_output) if config.normalize else slate_output
                                if slate_key not in recent_slate_outputs:
                                    if slate_file:
                                        slate_file.write(f"{slate_key}\n")
                                        slate_file.flush()
                                    recent_slate_outputs.append(slate_key)
                                    if len(recent_slate_outputs) > dedup_window_size:
                                        recent_slate_outputs.pop(0)
                                    last_written_slate = slate_key
                        else:
                            if scribe_output and scribe_file:
                                if config.normalize:
                                    scribe_output = normalize_text(scribe_output)
                                scribe_file.write(f"{scribe_output}\n")
                                scribe_file.flush()
                            if slate_output and slate_file:
                                if config.normalize:
                                    slate_output = normalize_text(slate_output)
                                slate_file.write(f"{slate_output}\n")
                                slate_file.flush()
            except Empty:
                if stop_flag.is_set():
                    log_drain_complete()
                    break
                idle_seconds += 1
                if not stream_thread.is_alive():
                    stream_ended = True
                    if config.verbose:
                        print("YouTube stream ended; draining buffer.")
                if stream_ended and idle_seconds >= 2:
                    break
                if not stream_ended and idle_seconds >= max_idle_seconds:
                    print("No audio received from YouTube stream; waiting for reconnect.")
                    idle_seconds = 0
                continue
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        stop_flag.set()
    finally:
        if len(buffer) > 0:
            if config.verbose:
                print("Processing final audio buffer. ..")
            try:
                result = process_audio_chunk(
                    buffer,
                    rate,
                    config.input_lang,
                    config.output_lang,
                    config.magnitude_threshold,
                    config.model,
                    config.verbose,
                    config.mqtt_broker,
                    config.mqtt_port,
                    config.mqtt_username,
                    config.mqtt_password,
                    config.mqtt_topic,
                    stream_id="youtube",
                    scribe_vad=config.scribe_vad,
                    voice_backend=config.voice_backend,
                    voice_model=config.voice_model,
                    timers=config.timers,
                    timing_stats=timing_stats,
                    scribe_backend=config.scribe_backend,
                    text_translation_target=config.text_translation_target,
                    slate_backend=config.slate_backend,
                    voice_lang=config.voice_lang,
                    scribe_text_file=None,
                    slate_text_file=None,
                    scribe_tts_segments=scribe_audio_segments,
                    slate_tts_segments=slate_audio_segments,
                    voice_match=config.voice_match,
                    lang_prefix=config.lang_prefix,
                )
                
                # Deduplication: Write outputs only if different from last ones
                if result:
                    scribe_output = result.get('scribe')
                    slate_output = result.get('slate')
                    if config.dedup:
                        if scribe_output:
                            scribe_key = normalize_text(scribe_output) if config.normalize else scribe_output
                            if scribe_key != last_written_scribe:
                                if scribe_file:
                                    scribe_file.write(f"{scribe_key}\n")
                                    scribe_file.flush()
                                last_written_scribe = scribe_key
      	       
                        if slate_output:
                            slate_key = normalize_text(slate_output) if config.normalize else slate_output
                            if slate_key != last_written_slate:
                                if slate_file:
                                    slate_file.write(f"{slate_key}\n")
                                    slate_file.flush()
                                last_written_slate = slate_key
                    else:
                        if scribe_output and scribe_file:
                            if config.normalize:
                                scribe_output = normalize_text(scribe_output)
                            scribe_file.write(f"{scribe_output}\n")
                            scribe_file.flush()
                        if slate_output and slate_file:
                            if config.normalize:
                                slate_output = normalize_text(slate_output)
                            slate_file.write(f"{slate_output}\n")
                            slate_file.flush()
            except Exception as exc:
                if config.verbose:
                    print(f"Final buffer processing failed: {exc}")
        if scribe_file:
            scribe_file.close()
            print(f"Stage 1 (English) text file saved: {config.scribe_text_file}", flush=True)
        if slate_file:
            slate_file.close()
            print(f"Stage 2 (translated) text file saved: {config.slate_text_file}", flush=True)
        if scribe_audio_segments is not None:
            if len(scribe_audio_segments) > 0:
                all_audio = np.concatenate(scribe_audio_segments)
                try:
                    output_audio(all_audio, config.output_audio_path, play=False)
                    print(f"Scribe audio file saved: {config.output_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving scribe audio file: {exc}", flush=True)
        if slate_audio_segments is not None:
            if len(slate_audio_segments) > 0:
                all_audio = np.concatenate(slate_audio_segments)
                try:
                    output_audio(all_audio, config.slate_audio_path, play=False)
                    print(f"Slate audio file saved: {config.slate_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving slate audio file: {exc}", flush=True)
        if capture_voice_segments is not None:
            if len(capture_voice_segments) > 0:
                all_audio = np.concatenate(capture_voice_segments)
                try:
                    output_audio(all_audio, config.capture_voice_path, play=False)
                    print(f"Captured input voice saved: {config.capture_voice_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving captured input voice: {exc}", flush=True)
        if timing_stats is not None:
            if config.timers_all:
                summary = timing_stats.format_summary()
                if summary:
                    print(f"\nTiming summary (youtube):\n{summary}")
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (youtube):\n{stage_summary}")
                overhead = timing_stats.format_translate_overhead('chunk')
                if overhead:
                    print(f"\nTiming translate overhead (youtube):\n{overhead}")
            elif config.timers:
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (youtube):\n{stage_summary}")