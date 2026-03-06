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

def run_realtime_youtube(
    youtube_url,
    youtube_api_key,
    input_lang=None,
    output_lang=None,
    # output_text_file removed
    magnitude_threshold=0.02,
    # play_audio removed
    output_audio_path=None,
    slate_audio_path=None,
    model=None,
    verbose=False,
    mqtt_broker=None,
    mqtt_port=1883,
    mqtt_username=None,
    mqtt_password=None,
    mqtt_topic="translation",
    scribe_vad=False,
    voice_backend="gtts",
    voice_model=None,
    youtube_js_runtime=None,
    youtube_remote_components=None,
    window_seconds=5.0,
    overlap_seconds=0.0,
    timers=False,
    timers_all=False,
    scribe_backend="auto",
    text_translation_target=None,
    slate_backend="googletrans",
    voice_lang=None,
    scribe_text_file=None,
    slate_text_file=None,
    voice_match=False,
    dedup=False,
    lang_prefix=False,
    normalize=True,
):
    print("Starting YouTube audio translation...")
    print(f"Input language: {input_lang}, Output language: {output_lang}")
    if output_audio_path:
        print(f"Output audio will be saved to: {output_audio_path}")
    # output_text_file removed
    if scribe_text_file:
        print(f"Stage 1 (English) text will be saved to: {scribe_text_file}")
    if slate_text_file:
        print(f"Stage 2 (translated) text will be saved to: {slate_text_file}")
    if mqtt_broker:
        print(f"MQTT output enabled: {mqtt_broker}:{mqtt_port}, topic: {mqtt_topic}")
    print("Press Ctrl+C to stop.")

    video_id = extract_youtube_video_id(youtube_url)
    if not video_id:
        print("Error: Unable to parse YouTube video ID from URL.")
        return

    validation = validate_youtube_video(youtube_api_key, video_id, verbose=verbose)
    if not validation:
        print("Error: YouTube API validation failed. Check API key and video ID.")
        return

    expected_duration = None
    content_details = validation.get("contentDetails") if isinstance(validation, dict) else None
    if content_details:
        expected_duration = parse_iso8601_duration(content_details.get("duration"))

    js_runtime = youtube_js_runtime
    if not js_runtime:
        if shutil.which("node"):
            js_runtime = "node"
        elif shutil.which("deno"):
            js_runtime = "deno"
    if verbose and js_runtime:
        print(f"Using yt-dlp JS runtime: {js_runtime}")

    if verbose and youtube_remote_components:
        print(f"Using yt-dlp remote components: {youtube_remote_components}")

    def resolve_audio_url():
        return get_youtube_audio_stream_url(
            youtube_url,
            verbose=verbose,
            js_runtime=js_runtime,
            remote_components=youtube_remote_components,
        )

    stop_flag = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping...", flush=True)
        stop_flag.set()

    signal.signal(signal.SIGINT, signal_handler)

    if mqtt_broker:
        init_mqtt(mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic)
    
    if timers_all:
        timers = True  # timers_all implies timers
        
    timing_stats = TimingsAggregator("youtube") if timers else None

    audio_queue = Queue(maxsize=5)
    buffer = np.array([], dtype=np.float32)
    # output_text_file removed
    scribe_file = open(scribe_text_file, mode="w", encoding="utf-8") if scribe_text_file else None
    slate_file = open(slate_text_file, mode="w", encoding="utf-8") if slate_text_file else None
    scribe_audio_segments = [] if output_audio_path else None
    slate_audio_segments = [] if slate_audio_path else None

    # Deduplication tracking: keep a sliding window of recent outputs to catch duplicates
    # across overlapping chunks.
    recent_slate_outputs = [] if dedup else None
    recent_scribe_outputs = [] if dedup else None
    dedup_window_size = 10  # Check last 10 outputs
    # Track last outputs to avoid duplicating the final buffer writes (see final buffer handling below).
    last_written_scribe = None
    last_written_slate = None

    stream_thread = threading.Thread(
        target=stream_youtube_audio,
        args=(resolve_audio_url, audio_queue, 16000, stop_flag, expected_duration, 5, verbose, False),
        daemon=True,
    )
    stream_thread.start()

    rate = 16000
    chunk, overlap = compute_window_params(window_seconds, overlap_seconds, rate)
    idle_seconds = 0
    max_idle_seconds = 60
    stream_ended = False

    try:
        while not stop_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=1)
                idle_seconds = 0
                buffer = np.concatenate([buffer, audio_chunk])
                if len(buffer) >= chunk:
                    audio_segment = buffer[:chunk]
                    buffer = buffer[chunk - overlap :]
                    result = process_audio_chunk(
                        audio_segment,
                        rate,
                        input_lang,
                        output_lang,
                        magnitude_threshold,
                        model,
                        verbose,
                        mqtt_broker,
                        mqtt_port,
                        mqtt_username,
                        mqtt_password,
                        mqtt_topic,
                        stream_id="youtube",
                        scribe_vad=scribe_vad,
                        voice_backend=voice_backend,
                        voice_model=voice_model,
                        timers=timers,
                        timing_stats=timing_stats,
                        scribe_backend=scribe_backend,
                        text_translation_target=text_translation_target,
                        slate_backend=slate_backend,
                        voice_lang=voice_lang,
                        scribe_text_file=None,
                        slate_text_file=None,
                        scribe_tts_segments=scribe_audio_segments,
                        slate_tts_segments=slate_audio_segments,
                        voice_match=voice_match,
                        lang_prefix=lang_prefix,
                    )
                    
                    # Deduplication: Write outputs only if not in recent window
                    if result:
                        scribe_output = result.get('scribe')
                        slate_output = result.get('slate')

                        if dedup:
                            if scribe_output and scribe_output not in recent_scribe_outputs:
                                if scribe_file:
                                    if normalize:
                                        scribe_output = normalize_text(scribe_output)
                                    scribe_file.write(f"{scribe_output}\n")
                                    scribe_file.flush()
                                recent_scribe_outputs.append(scribe_output)
                                if len(recent_scribe_outputs) > dedup_window_size:
                                    recent_scribe_outputs.pop(0)
                                last_written_scribe = scribe_output

                            if slate_output and slate_output not in recent_slate_outputs:
                                if slate_file:
                                    if normalize:
                                        slate_output = normalize_text(slate_output)
                                    slate_file.write(f"{slate_output}\n")
                                    slate_file.flush()
                                recent_slate_outputs.append(slate_output)
                                if len(recent_slate_outputs) > dedup_window_size:
                                    recent_slate_outputs.pop(0)
                                last_written_slate = slate_output
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
            except Empty:
                idle_seconds += 1
                if not stream_thread.is_alive():
                    stream_ended = True
                    if verbose:
                        print("YouTube stream ended; draining buffer.")
                if stream_ended and idle_seconds >= 2:
                    break
                if not stream_ended and idle_seconds >= max_idle_seconds:
                    print("No audio received from YouTube stream; waiting for reconnect.")
                    idle_seconds = 0
                if stop_flag.is_set():
                    break
                continue
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        stop_flag.set()
    finally:
        if len(buffer) > 0:
            if verbose:
                print("Processing final audio buffer...")
            try:
                result = process_audio_chunk(
                    buffer,
                    rate,
                    input_lang,
                    output_lang,
                    magnitude_threshold,
                    model,
                    verbose,
                    mqtt_broker,
                    mqtt_port,
                    mqtt_username,
                    mqtt_password,
                    mqtt_topic,
                    stream_id="youtube",
                    scribe_vad=scribe_vad,
                    voice_backend=voice_backend,
                    voice_model=voice_model,
                    timers=timers,
                    timing_stats=timing_stats,
                    scribe_backend=scribe_backend,
                    text_translation_target=text_translation_target,
                    slate_backend=slate_backend,
                    voice_lang=voice_lang,
                    scribe_text_file=None,
                    slate_text_file=None,
                    scribe_tts_segments=scribe_audio_segments,
                    slate_tts_segments=slate_audio_segments,
                    voice_match=voice_match,
                    lang_prefix=lang_prefix,
                )
                
                # Deduplication: Write outputs only if different from last ones
                if result:
                    scribe_output = result.get('scribe')
                    slate_output = result.get('slate')
                    if dedup:
                        if scribe_output and scribe_output != last_written_scribe:
                            if scribe_file:
                                if normalize:
                                    scribe_output = normalize_text(scribe_output)
                                scribe_file.write(f"{scribe_output}\n")
                                scribe_file.flush()
                            last_written_scribe = scribe_output
                     
                        if slate_output and slate_output != last_written_slate:
                            if slate_file:
                                if normalize:
                                    slate_output = normalize_text(slate_output)
                                slate_file.write(f"{slate_output}\n")
                                slate_file.flush()
                            last_written_slate = slate_output
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
            except Exception as exc:
                if verbose:
                    print(f"Final buffer processing failed: {exc}")
        if scribe_file:
            scribe_file.close()
            print(f"Stage 1 (English) text file saved: {scribe_text_file}", flush=True)
        if slate_file:
            slate_file.close()
            print(f"Stage 2 (translated) text file saved: {slate_text_file}", flush=True)
        if scribe_audio_segments is not None:
            if len(scribe_audio_segments) > 0:
                all_audio = np.concatenate(scribe_audio_segments)
                try:
                    output_audio(all_audio, output_audio_path, play=False)
                    print(f"Scribe audio file saved: {output_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving scribe audio file: {exc}", flush=True)
        if slate_audio_segments is not None:
            if len(slate_audio_segments) > 0:
                all_audio = np.concatenate(slate_audio_segments)
                try:
                    output_audio(all_audio, slate_audio_path, play=False)
                    print(f"Slate audio file saved: {slate_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving slate audio file: {exc}", flush=True)
        if timing_stats is not None:
            if timers_all:
                summary = timing_stats.format_summary()
                if summary:
                    print(f"\nTiming summary (youtube):\n{summary}")
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (youtube):\n{stage_summary}")
                overhead = timing_stats.format_translate_overhead('chunk')
                if overhead:
                    print(f"\nTiming translate overhead (youtube):\n{overhead}")
            elif timers:
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (youtube):\n{stage_summary}")
