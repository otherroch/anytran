import json
import re
import shutil
import urllib.request

import numpy as np


def extract_youtube_video_id(youtube_url):
    try:
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(youtube_url)
        if parsed.hostname in {"youtu.be"}:
            return parsed.path.lstrip("/")
        if parsed.hostname and "youtube.com" in parsed.hostname:
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]
            if parsed.path.startswith("/shorts/"):
                return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
            if parsed.path.startswith("/live/"):
                return parsed.path.split("/live/", 1)[1].split("/", 1)[0]
        return None
    except Exception:
        return None


def validate_youtube_video(api_key, video_id, verbose=False):
    if not api_key or not video_id:
        return None

    api_url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,contentDetails,liveStreamingDetails&id={video_id}&key={api_key}"
    )
    try:
        with urllib.request.urlopen(api_url, timeout=10) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        items = data.get("items") or []
        if not items:
            return None
        item = items[0]
        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})
        live_details = item.get("liveStreamingDetails", {})
        if verbose:
            print(f"YouTube video validated: {snippet.get('title', 'unknown title')}")
        return {
            "snippet": snippet,
            "contentDetails": content_details,
            "liveStreamingDetails": live_details,
        }
    except Exception as exc:
        if verbose:
            print(f"YouTube API validation failed: {exc}")
        return None


def get_youtube_audio_stream_url(youtube_url, verbose=False, js_runtime=None, remote_components=None):
    try:
        from yt_dlp import YoutubeDL
    except Exception as e:
        print(f"yt-dlp not installed or import failed: {e}")
        print("Install with: pip install yt-dlp")
        return None

    ydl_opts = {
        "quiet": not verbose,
        "no_warnings": not verbose,
        "skip_download": True,
        "format": "bestaudio/best",
        "noplaylist": True,
    }

    if js_runtime:
        runtime_name = js_runtime
        runtime_path = None
        if ":" in js_runtime:
            runtime_name, runtime_path = js_runtime.split(":", 1)
        runtime_name = runtime_name.strip()
        if runtime_path:
            runtime_path = runtime_path.strip()
            ydl_opts["js_runtimes"] = {runtime_name: {"path": runtime_path}}
        else:
            ydl_opts["js_runtimes"] = {runtime_name: {}}

    if remote_components:
        ydl_opts["remote_components"] = [remote_components]

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        if not info:
            return None
        if "url" in info:
            return info.get("url")
        formats = info.get("formats") or []
        audio_only = [f for f in formats if f.get("vcodec") == "none" and f.get("url")]
        if audio_only:
            return audio_only[-1].get("url")
        for fmt in reversed(formats):
            if fmt.get("url"):
                return fmt.get("url")
        return None
    except Exception as exc:
        print(f"yt-dlp failed to resolve audio stream: {exc}")
        return None


def parse_iso8601_duration(duration_text):
    if not duration_text:
        return None
    match = re.match(r"^P(T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)$", duration_text)
    if not match:
        return None
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    return hours * 3600 + minutes * 60 + seconds


def stream_youtube_audio(resolve_audio_url, audio_queue, sample_rate=16000, stop_event=None, expected_duration=None, max_retries=5, verbose=False, drop_if_full=True):
    try:
        import av

        options = {
            "reconnect": "1",
            "reconnect_streamed": "1",
            "reconnect_delay_max": "5",
        }

        total_samples = 0
        retries = 0
        while stop_event is None or not stop_event.is_set():
            audio_url = resolve_audio_url()
            if not audio_url:
                print("Unable to resolve YouTube audio stream URL.")
                return
            if verbose:
                print(f"Resolved YouTube audio URL (len={len(audio_url)}): {audio_url[:120]}...")

            container = av.open(audio_url, options=options)
            audio_stream = None
            for stream in container.streams:
                if stream.type == "audio":
                    audio_stream = stream
                    break

            if audio_stream is None:
                print("No audio stream found for YouTube URL")
                return

            if total_samples > 0:
                try:
                    seek_time = total_samples / float(sample_rate)
                    container.seek(int(seek_time * av.time_base), stream=audio_stream)
                except Exception:
                    if verbose:
                        print("Seek not supported; continuing from stream start.")
            if verbose:
                print(
                    "Starting decode loop (retry {0}, processed {1:.2f}s)".format(
                        retries, total_samples / float(sample_rate)
                    )
                )

            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=sample_rate,
            )

            try:
                for frame in container.decode(audio=0):
                    if stop_event is not None and stop_event.is_set():
                        break
                    resampled_frames = resampler.resample(frame)
                    for resampled_frame in resampled_frames:
                        audio_array = resampled_frame.to_ndarray()
                        audio_np = audio_array.astype(np.float32) / 32768.0
                        if audio_np.ndim > 1:
                            audio_np = audio_np.flatten()
                        total_samples += len(audio_np)
                        if drop_if_full and audio_queue.full():
                            try:
                                audio_queue.get_nowait()
                            except Exception:
                                pass
                        if drop_if_full:
                            audio_queue.put(audio_np)
                        else:
                            while True:
                                if stop_event is not None and stop_event.is_set():
                                    break
                                try:
                                    audio_queue.put(audio_np, timeout=1)
                                    break
                                except Exception:
                                    continue
            except Exception as exc:
                if verbose:
                    print(f"YouTube stream decode error: {exc}")
            finally:
                try:
                    container.close()
                except Exception:
                    pass
                if verbose:
                    print("YouTube stream decode loop ended.")

            if expected_duration:
                processed_seconds = total_samples / float(sample_rate)
                if processed_seconds >= expected_duration - 1:
                    if verbose:
                        print(f"Reached expected duration ({processed_seconds:.2f}s).")
                    break

            retries += 1
            if retries > max_retries:
                if verbose:
                    print("Max YouTube reconnect attempts reached.")
                break
            if verbose:
                print("YouTube stream ended early; reconnecting...")
    except ImportError:
        print("PyAV not installed. Please install with: pip install av")
        return
    except Exception as exc:
        print(f"Error streaming YouTube audio: {exc}")
        import traceback

        traceback.print_exc()
