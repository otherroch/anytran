from anytran.audio_io import output_audio
from anytran.chatlog import ChatLogger, extract_ip_from_rtsp_url
from anytran.mqtt_client import init_mqtt
from anytran.normalizer import normalize_text
from anytran.processing import process_audio_chunk
from anytran.stream_rtsp import stream_rtsp_audio
from anytran.timing import TimingsAggregator
from anytran.utils import compute_window_params
import threading
import numpy as np
import signal
import time
from queue import Queue, Empty

def run_multi_rtsp(
    rtsp_urls,
    input_lang=None,
    output_lang=None,
    output_audio_path=None,
    slate_audio_path=None,
    # output_text_file removed
    magnitude_threshold=0.02,
    # play_audio removed
    model=None,
    verbose=False,
    mqtt_broker=None,
    mqtt_port=1883,
    mqtt_username=None,
    mqtt_password=None,
    mqtt_topic="translation",
    topic_names=None,
    scribe_vad=False,
    voice_backend="gtts",
    voice_model=None,
    chat_log_dir=None,
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
    print(f"Starting {len(rtsp_urls)} RTSP streams...")

    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping all streams...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    chat_logger = None
    if chat_log_dir:
        chat_logger = ChatLogger(chat_log_dir)
        print(f"Chat logging enabled. Logs will be saved to: {chat_log_dir}")

    if mqtt_broker:
        if topic_names and len(topic_names) == len(rtsp_urls):
            print(f"MQTT output enabled: {mqtt_broker}:{mqtt_port}")
            for i, topic in enumerate(topic_names, 1):
                print(f"  Stream {i} -> topic: {topic}")
        else:
            print(f"MQTT output enabled: {mqtt_broker}:{mqtt_port}")
            for i in range(len(rtsp_urls)):
                print(f"  Stream {i + 1} -> topic: stream{i + 1}")
        init_mqtt(mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic)
    if scribe_text_file:
        print(f"Scribe text (English) will be saved to: {scribe_text_file}")
    if slate_text_file:
        print(f"Slate text (translated) will be saved to: {slate_text_file}")
    # output_text_file removed
    scribe_file = open(scribe_text_file, mode="w", encoding="utf-8") if scribe_text_file else None
    slate_file = open(slate_text_file, mode="w", encoding="utf-8") if slate_text_file else None
    if timers_all:
        timers = True  # timers_all implies timers       
    timing_stats = TimingsAggregator("multi_rtsp") if timers else None

    def worker(rtsp_url, idx):
        # Deduplication tracking for this worker's stream
        last_scribe_output = None
        last_slate_output = None
        recent_slate_outputs = []
        recent_scribe_outputs = []
        dedup_window_size = 10  # Check last 10 outputs
        audio_queue = Queue(maxsize=5)
        buffer = np.array([], dtype=np.float32)
        stream_thread = threading.Thread(target=stream_rtsp_audio, args=(rtsp_url, audio_queue), daemon=True)
        stream_thread.start()
        rate = 16000
        chunk, overlap = compute_window_params(window_seconds, overlap_seconds, rate)
        local_scribe_audio_segments = [] if output_audio_path else None
        local_slate_audio_segments = [] if slate_audio_path else None
        rtsp_ip = extract_ip_from_rtsp_url(rtsp_url) if chat_logger else None
        if chat_logger and rtsp_ip:
            print(f"[Stream {idx}] RTSP IP: {rtsp_ip}")
        if topic_names and len(topic_names) >= idx:
            stream_mqtt_topic = topic_names[idx - 1]
        else:
            stream_mqtt_topic = f"stream{idx}"

        try:
            while not stop_event.is_set():
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
                            stream_mqtt_topic,
                            stream_id=idx,
                            scribe_vad=scribe_vad,
                            voice_backend=voice_backend,
                            voice_model=voice_model,
                            chat_logger=chat_logger,
                            rtsp_ip=rtsp_ip,
                            timers=timers,
                            timing_stats=timing_stats,
                            scribe_backend=scribe_backend,
                            text_translation_target=text_translation_target,
                            slate_backend=slate_backend,
                            voice_lang=voice_lang,
                            scribe_text_file=None,
                            slate_text_file=None,
                            scribe_tts_segments=local_scribe_audio_segments,
                            slate_tts_segments=local_slate_audio_segments,
                            voice_match=voice_match,
                            lang_prefix=lang_prefix,
                        )
                        # Deduplication: Write outputs only if not in recent window (if dedup enabled)
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
                    if stop_event.is_set():
                        break
                    continue
        except KeyboardInterrupt:
            if verbose:
                print(f"[Stream {idx}] Stopped.")
        finally:
            if local_scribe_audio_segments is not None:
                if len(local_scribe_audio_segments) > 0:
                    all_audio = np.concatenate(local_scribe_audio_segments)
                    output_file = (
                        f"{output_audio_path.rsplit('.', 1)[0]}_stream{idx}.{output_audio_path.rsplit('.', 1)[1]}"
                        if output_audio_path
                        else None
                    )
                    if output_file:
                        try:
                            output_audio(all_audio, output_file, play=False)
                            print(f"[Stream {idx}] Scribe audio saved to {output_file}", flush=True)
                        except Exception as exc:
                            print(f"[Stream {idx}] Error saving scribe audio file: {exc}", flush=True)
                            import traceback
                            traceback.print_exc()
            if local_slate_audio_segments is not None:
                if len(local_slate_audio_segments) > 0:
                    all_audio = np.concatenate(local_slate_audio_segments)
                    output_file = (
                        f"{slate_audio_path.rsplit('.', 1)[0]}_stream{idx}.{slate_audio_path.rsplit('.', 1)[1]}"
                        if slate_audio_path
                        else None
                    )
                    if output_file:
                        try:
                            output_audio(all_audio, output_file, play=False)
                            print(f"[Stream {idx}] Slate audio saved to {output_file}", flush=True)
                        except Exception as exc:
                            print(f"[Stream {idx}] Error saving slate audio file: {exc}", flush=True)
                            import traceback
                            traceback.print_exc()

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
        if text_file:
            text_file.close()
        if scribe_file:
            scribe_file.close()
            print(f"Scribe text file saved: {scribe_text_file}", flush=True)
        if slate_file:
            slate_file.close()
            print(f"Slate text file saved: {slate_text_file}", flush=True)
        if timing_stats is not None:
            if timers_all:
                summary = timing_stats.format_summary()
                if summary:
                    print(f"\nTiming summary (multi-rtsp):\n{summary}")
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (multi-rtsp):\n{stage_summary}")
                overhead = timing_stats.format_translate_overhead('chunk')
                if overhead:
                    print(f"\nTiming translate overhead (multi-rtsp):\n{overhead}")
            elif timers:
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (multi-rtsp):\n{stage_summary}")
