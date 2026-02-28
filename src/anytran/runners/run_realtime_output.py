from anytran.stream_output import get_wasapi_loopback_device_info, stream_output_audio
from anytran.audio_io import output_audio
from anytran.normalizer import normalize_text
from anytran.processing import process_audio_chunk
from anytran.mqtt_client import init_mqtt
from anytran.timing import TimingsAggregator
from anytran.utils import compute_window_params
import threading
import numpy as np
import signal
import sys
from queue import Queue, Empty

def run_realtime_output(
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
    output_device=None,
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
    print("Starting real-time system output audio translation...")
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

    if sys.platform != "win32":
        print("System output capture is only supported on Windows (WASAPI loopback).")
        return

    device_info = get_wasapi_loopback_device_info(preferred_name=output_device, verbose=verbose)
    if not device_info:
        print("No WASAPI loopback device found. Unable to capture system output.")
        return

    stop_flag = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping...", flush=True)
        stop_flag.set()

    signal.signal(signal.SIGINT, signal_handler)

    if mqtt_broker:
        init_mqtt(mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic)

    if timers_all:
        timers = True  # timers_all implies timers      
    timing_stats = TimingsAggregator("output") if timers else None
 
    audio_queue = Queue(maxsize=5)
    buffer = np.array([], dtype=np.float32)
    # output_text_file removed
    scribe_file = open(scribe_text_file, mode="w", encoding="utf-8") if scribe_text_file else None
    slate_file = open(slate_text_file, mode="w", encoding="utf-8") if slate_text_file else None
    scribe_audio_segments = [] if output_audio_path else None
    slate_audio_segments = [] if slate_audio_path else None

    # Deduplication tracking for dual text output

    recent_slate_outputs = []
    recent_scribe_outputs = []
    dedup_window_size = 10  # Check last 10 outputs

    stream_thread = threading.Thread(
        target=stream_output_audio,
        args=(audio_queue, device_info, 16000, stop_flag, verbose),
        daemon=True,
    )
    stream_thread.start()

    rate = 16000
    chunk, overlap = compute_window_params(window_seconds, overlap_seconds, rate)

    try:
        while not stop_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=1)
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
                        stream_id="output",
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

                            if slate_output and slate_output not in recent_slate_outputs:
                                if slate_file:
                                    if normalize:
                                        slate_output = normalize_text(slate_output)
                                    slate_file.write(f"{slate_output}\n")
                                    slate_file.flush()
                                recent_slate_outputs.append(slate_output)
                                if len(recent_slate_outputs) > dedup_window_size:
                                    recent_slate_outputs.pop(0)
                        else:
                            # When deduplication is disabled, write outputs directly
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
                if stop_flag.is_set():
                    break
                continue
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        stop_flag.set()
    finally:
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
                    print(f"\nTiming summary (output):\n{summary}")
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (output):\n{stage_summary}")
                overhead = timing_stats.format_translate_overhead('chunk')
                if overhead:
                    print(f"\nTiming translate overhead (output):\n{overhead}")
            elif timers:
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (output):\n{stage_summary}")
