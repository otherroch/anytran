from anytran.audio_io import load_audio_any, output_audio
from anytran.chatlog import ChatLogger, extract_ip_from_rtsp_url
from anytran.mqtt_client import init_mqtt, send_mqtt_text
from anytran.normalizer import normalize_text
from anytran.processing import build_output_prefix, process_audio_chunk
from anytran.stream_output import get_wasapi_loopback_device_info, stream_output_audio
from anytran.stream_rtsp import stream_rtsp_audio
from anytran.stream_youtube import (
    extract_youtube_video_id,
    get_youtube_audio_stream_url,
    parse_iso8601_duration,
    stream_youtube_audio,
    validate_youtube_video,
)
from anytran.text_translator import translate_text
from anytran.config import get_whisper_backend
from anytran.timing import TimingsAggregator, add_timing
from anytran.tts import play_output, synthesize_tts_pcm
from anytran.utils import compute_window_params
from .config import RunnerConfig
import threading
import numpy as np
import signal
import time
from queue import Queue, Empty

def run_realtime_rtsp(config):
    # Create a RunnerConfig instance from parameters if needed
    if not isinstance(config, RunnerConfig):
        # If config is not a RunnerConfig, create one from the parameters
        config = RunnerConfig(**config)
    
    print("Starting real-time RTSP audio translation...")
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

    stop_flag = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping...", flush=True)
        stop_flag.set()

    signal.signal(signal.SIGINT, signal_handler)

    chat_logger = None
    rtsp_ip = None
    if config.chat_log_dir:
        chat_logger = ChatLogger(config.chat_log_dir)
        rtsp_ip = extract_ip_from_rtsp_url(config.rtsp_url)
        print(f"Chat logging enabled. Logs will be saved to: {config.chat_log_dir}")
        print(f"RTSP IP: {rtsp_ip}")

    if config.mqtt_broker:
        init_mqtt(config.mqtt_broker, config.mqtt_port, config.mqtt_username, config.mqtt_password, config.mqtt_topic)

    if config.timers_all:
        config.timers = True  # timers_all implies timers       
    timing_stats = TimingsAggregator("rtsp") if config.timers else None

    audio_queue = Queue(maxsize=50)
    buffer = np.array([], dtype=np.float32)
    # output_text_file removed
    scribe_file = open(config.scribe_text_file, mode="w", encoding="utf-8") if config.scribe_text_file else None
    slate_file = open(config.slate_text_file, mode="w", encoding="utf-8") if config.slate_text_file else None
    scribe_audio_segments = [] if config.output_audio_path else None
    slate_audio_segments = [] if config.slate_audio_path else None
    capture_voice_segments = [] if config.capture_voice_path else None

    # Deduplication tracking for dual text output
    last_scribe_output = None
    last_slate_output = None
    recent_slate_outputs = []
    recent_scribe_outputs = []
    dedup_window_size = 10  # Check last 10 outputs

    stream_thread = threading.Thread(
        target=stream_rtsp_audio,
        args=(config.rtsp_url, audio_queue),
        daemon=True,
    )
    stream_thread.start()
    rate = 16000
    chunk, overlap = compute_window_params(config.window_seconds, config.overlap_seconds, rate)
    try:
        while not stop_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=1)
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
                        scribe_vad=config.scribe_vad,
                        voice_backend=config.voice_backend,
                        voice_model=config.voice_model,
                        chat_logger=chat_logger,
                        rtsp_ip=rtsp_ip,
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
                    # Deduplication: Write outputs only if not in recent window (if dedup enabled)
                    if result:
                        scribe_output = result.get('scribe')
                        slate_output = result.get('slate')
                        if config.dedup:
                            if scribe_output and scribe_output not in recent_scribe_outputs:
                                if scribe_file:
                                    if config.normalize:
                                        scribe_output = normalize_text(scribe_output)
                                    scribe_file.write(f"{scribe_output}\n")
                                    scribe_file.flush()
                                recent_scribe_outputs.append(scribe_output)
                                if len(recent_scribe_outputs) > dedup_window_size:
                                    recent_scribe_outputs.pop(0)
                            if slate_output and slate_output not in recent_slate_outputs:
                                if slate_file:
                                    if config.normalize:
                                        slate_output = normalize_text(slate_output)
                                    slate_file.write(f"{slate_output}\n")
                                    slate_file.flush()
                                recent_slate_outputs.append(slate_output)
                                if len(recent_slate_outputs) > dedup_window_size:
                                    recent_slate_outputs.pop(0)
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
                if stop_flag.is_set():
                    break
                print(f"Error during realtime RTSP processing loop: {exc}", flush=True)
                continue
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
        # output_text_file removed
        if scribe_audio_segments is not None:
            if len(scribe_audio_segments) > 0:
                all_audio = np.concatenate(scribe_audio_segments)
                try:
                    output_audio(all_audio, config.output_audio_path, play=False)
                    print(f"Scribe audio file saved: {config.output_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving scribe audio file: {exc}", flush=True)
                    import traceback

                    traceback.print_exc()
        if slate_audio_segments is not None:
            if len(slate_audio_segments) > 0:
                all_audio = np.concatenate(slate_audio_segments)
                try:
                    output_audio(all_audio, config.slate_audio_path, play=False)
                    print(f"Slate audio file saved: {config.slate_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving slate audio file: {exc}", flush=True)
                    import traceback

                    traceback.print_exc()
        if capture_voice_segments is not None:
            if len(capture_voice_segments) > 0:
                all_audio = np.concatenate(capture_voice_segments)
                try:
                    output_audio(all_audio, config.capture_voice_path, play=False)
                    print(f"Captured input voice saved: {config.capture_voice_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving captured input voice: {exc}", flush=True)
                    import traceback

                    traceback.print_exc()
        if timing_stats is not None:
            stage_summary = timing_stats.format_stage_summary()
            if stage_summary:
                print(f"\nTiming summary (rtsp):\n{stage_summary}")
