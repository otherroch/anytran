import atexit
import os
import re
import sys


def restore_terminal():
    if sys.platform != "win32":
        try:
            os.system("stty sane")
        except Exception:
            pass


atexit.register(restore_terminal)


def normalize_lang_code(lang_code):
    if not lang_code:
        return None
    return str(lang_code).strip().lower()


def compute_window_params(window_seconds, overlap_seconds, rate):
    window_samples = max(1, int(window_seconds * rate))
    overlap_samples = max(0, int(overlap_seconds * rate))
    if overlap_samples >= window_samples:
        overlap_samples = max(0, window_samples - 1)
    return window_samples, overlap_samples


def extract_ip_from_rtsp_url(rtsp_url):
    try:
        from urllib.parse import urlparse

        parsed = urlparse(rtsp_url)
        host = parsed.hostname or parsed.netloc.split(":")[0].split("@")[-1]
        return host
    except Exception:
        match = re.search(r"//([^:/]+)", rtsp_url or "")
        if match:
            return match.group(1)
        return "unknown"


def resolve_path_with_fallback(path, fallback_dir):
    if not path:
        return None
    # Expand ~ to home directory first
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(fallback_dir, path)
    return candidate if os.path.exists(candidate) else path
