import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import librosa
import numpy as np
import soundfile as sf

from .config import (
    get_whisper_backend,
    get_whisper_cpp_config,
    get_whispercpp_cli_detect_lang,
    get_whispercpp_force_cli,
    get_whisper_ctranslate2_config,
)
from .timing import add_timing, format_timing
from .utils import normalize_lang_code

try:
    from whisper_ctranslate2.transcribe import Transcribe, TranscriptionOptions
    _whisper_ctranslate2_available = True
    _whisper_ctranslate2_import_error = None
except ImportError as _whisper_ctranslate2_exc:
    Transcribe = None
    TranscriptionOptions = None
    _whisper_ctranslate2_available = False
    _whisper_ctranslate2_import_error = _whisper_ctranslate2_exc
WHISPER_CPP_MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

_whispercpp_model = None
_whispercpp_model_id = None
_whispercpp_binding_available = True
_whispercpp_binding_error = None

_whisper_model = None
_whisper_device = None

_whisper_ctranslate2_model = None
_whisper_ctranslate2_model_id = None
_whisper_ctranslate2_device = None

HALLUCINATION_PHRASES = [
    "thank you for watching",
    "please subscribe",
    "welcome to my channel",
    "press the bell icon",
    "like and subscribe",
    "don't forget to subscribe",
    "see you in the next video",
    "thanks for watching",
    "please like",
    "leave a comment",
    "watch the video until the end",
    "this is the end of the video",
    "thank you for your viewing",
]


def _get_with_override(override_value, config_dict, config_key, default=None):
    """
    Resolve a configuration value, preferring an explicit override over
    values defined in a configuration dictionary.

    This helper is used to centralize the common pattern for configuration
    lookup: if a caller provides an explicit value, that value wins; otherwise
    the value is read from a configuration mapping, falling back to a default
    when the key is absent.

    Args:
        override_value: A value explicitly provided by the caller. If this is
            not ``None``, it is returned as-is and the configuration mapping is
            not consulted.
        config_dict: A mapping (typically a configuration dictionary) from
            which to read the value when no explicit override is provided.
        config_key: The key used to look up the value in ``config_dict`` when
            ``override_value`` is ``None``.
        default: The value to return if ``override_value`` is ``None`` and
            ``config_key`` is not present in ``config_dict``. Defaults to
            ``None``.

    Returns:
        The resolved configuration value: ``override_value`` if it is not
        ``None``, otherwise the value from ``config_dict`` for ``config_key``,
        or ``default`` if the key is missing.
    """
    if override_value is not None:
        return override_value
    return config_dict.get(config_key, default)


def is_hallucination(text):
    if not text or len(text.strip()) < 5:
        return True

    text_lower = text.lower()

    for phrase in HALLUCINATION_PHRASES:
        if phrase in text_lower:
            return True

    words = text_lower.split()
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            return True

    return False


def _derive_whispercpp_model_name(model_path):
    if not model_path:
        return None
    base = os.path.basename(model_path)
    if base.startswith("ggml-") and base.endswith(".bin"):
        return base[len("ggml-") : -len(".bin")]
    name = os.path.splitext(base)[0]
    known = {"tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v1", "large"}
    return name if name in known else None


def get_whispercpp_model(model_path, verbose=False):
    global _whispercpp_model, _whispercpp_model_id, _whispercpp_binding_available, _whispercpp_binding_error
    if not _whispercpp_binding_available:
        raise RuntimeError(_whispercpp_binding_error or "pywhispercpp binding unavailable")
    model_id = model_path
    if _whispercpp_model is not None and _whispercpp_model_id == model_id:
        return _whispercpp_model

    try:
        from pywhispercpp.model import Model
    except Exception as exc:
        _whispercpp_binding_available = False
        _whispercpp_binding_error = "pywhispercpp not installed. Install with: pip install pywhispercpp"
        raise RuntimeError(_whispercpp_binding_error) from exc

    if not getattr(Model, "_safe_del_installed", False):
        original_del = getattr(Model, "__del__", None)

        def _safe_model_del(self):
            try:
                if callable(original_del):
                    original_del(self)
            except TypeError:
                pass
            except Exception:
                pass

        Model.__del__ = _safe_model_del
        Model._safe_del_installed = True

    derived_name = _derive_whispercpp_model_name(model_path)
    if verbose:
        print(f"Loading pywhispercpp model from: {model_path}")
        if derived_name and derived_name != model_path:
            print(f"Derived pywhispercpp model name: {derived_name}")

    try:
        _whispercpp_model = Model(model_path, print_realtime=False, print_progress=False)
        _whispercpp_model_id = model_id
        return _whispercpp_model
    except Exception as exc:
        if not derived_name:
            raise
        if verbose:
            print(f"pywhispercpp fallback to model name '{derived_name}' due to: {exc}")
        _whispercpp_model = Model(derived_name, print_realtime=False, print_progress=False)
        _whispercpp_model_id = derived_name
        return _whispercpp_model


def get_faster_whisper_model(model_size="medium", device_preference="cuda", compute_type="float16"):
    global _whisper_model, _whisper_device

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError("faster-whisper not installed. Install with: pip install faster-whisper") from exc

    use_cuda = False
    if device_preference in ("cuda", "auto"):
        try:
            import ctranslate2

            use_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            use_cuda = False
        if not use_cuda:
            try:
                import torch

                use_cuda = torch.cuda.is_available()
            except Exception:
                use_cuda = False

    if device_preference == "cpu":
        use_cuda = False

    device = "cuda" if use_cuda else "cpu"
    model_key = (model_size, device, compute_type)
    if _whisper_model is not None and _whisper_device == model_key:
        return _whisper_model
    _whisper_device = model_key
    print(f"Loading faster-whisper model '{model_size}' on device: {device} with compute_type: {compute_type}...")
    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _whisper_model


def _resolve_faster_whisper_model_path(model_size, model_obj=None):
    if model_size and os.path.exists(model_size):
        return os.path.abspath(model_size)

    for attr in ("model_dir", "model_path", "model_folder", "path"):
        value = getattr(model_obj, attr, None)
        if isinstance(value, str) and value:
            return value

    repo_id = None
    if model_size and "/" in model_size:
        repo_id = model_size
    elif model_size:
        repo_id = f"Systran/faster-whisper-{model_size}"

    if repo_id:
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(repo_id=repo_id, local_files_only=True)
        except Exception:
            return None

    return None


def _resolve_ctranslate2_model_path(model_name, model_obj=None):
    if model_name and os.path.exists(model_name):
        return os.path.abspath(model_name)

    for attr in ("model_dir", "model_path", "model_folder", "path"):
        value = getattr(model_obj, attr, None)
        if isinstance(value, str) and value:
            return value

    if model_name and "/" in model_name:
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(repo_id=model_name, local_files_only=True)
        except Exception:
            return None

    if model_name:
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            matches = []
            suffix = f"-{model_name}"
            for repo in cache_info.repos:
                repo_id = getattr(repo, "repo_id", "")
                repo_id_lower = repo_id.lower()
                if "whisper" not in repo_id_lower:
                    continue
                if "faster-whisper" in repo_id_lower:
                    continue
                if model_name not in repo_id_lower and not repo_id_lower.endswith(suffix):
                    continue
                matches.append(repo)
            if matches:
                matches.sort(key=lambda repo: repo.repo_id)
                return matches[0].repo_path
        except Exception:
            return None

    return None


def get_whisper_ctranslate2_model(model_name="small", device="auto", device_index=None, compute_type="default", verbose=False):
    """
    Load and cache a whisper-ctranslate2 Transcribe model with the given configuration.

    This helper chooses an appropriate device (CPU or CUDA) when ``device="auto"``,
    instantiates the underlying :class:`whisper_ctranslate2.transcribe.Transcribe`
    object, and caches it so that subsequent calls with the same parameters reuse
    the existing model instance.

    Parameters
    ----------
    model_name : str, optional
        Name or path of the whisper-ctranslate2 model to load. Defaults to ``"small"``.
    device : str, optional
        Device on which to load the model (e.g. ``"cuda"`` or ``"cpu"``). If set to
        ``"auto"``, CUDA will be used when available, otherwise CPU. Defaults to ``"auto"``.
    device_index : int or None, optional
        Index of the device to use when running on CUDA. If ``None``, the default
        device is used. Defaults to ``None``.
    compute_type : str, optional
        Compute type passed through to whisper-ctranslate2 (e.g. precision or quantization
        configuration). Defaults to ``"default"``.
    verbose : bool, optional
        If ``True``, prints information about model loading and cache reuse. Defaults
        to ``False``.

    Returns
    -------
    Transcribe
        A configured and possibly cached instance of
        :class:`whisper_ctranslate2.transcribe.Transcribe`.

    Raises
    ------
    RuntimeError
        If the ``whisper-ctranslate2`` package is not installed.
    """
    global _whisper_ctranslate2_model, _whisper_ctranslate2_model_id, _whisper_ctranslate2_device

    # Validate that whisper-ctranslate2 is installed
    if not _whisper_ctranslate2_available:
        raise RuntimeError(
            f"whisper-ctranslate2 package is not installed. "
            f"Install it with: pip install whisper-ctranslate2 ctranslate2. "
            f"Original error: {_whisper_ctranslate2_import_error}"
        )

    # Determine device
    if device == "auto":
        device = "cuda"
        try:
            import ctranslate2
            device_count = ctranslate2.get_cuda_device_count()
            if device_count == 0:
                device = "cpu"
        except Exception:
            device = "cpu"

    # Default device_index to 0 if None
    _device_index = device_index if device_index is not None else 0
    
    model_key = (model_name, device, device_index, compute_type)
    if _whisper_ctranslate2_model is not None and _whisper_ctranslate2_model_id == model_key:
        if verbose:
            print(f"Reusing cached whisper-ctranslate2 model for key: {model_key}")
        return _whisper_ctranslate2_model

    if verbose:
        print(f"Loading whisper-ctranslate2 model '{model_name}' on device: {device}", end="")
        if device_index is not None:
            print(f" (index: {device_index})", end="")
        if compute_type and compute_type != "default":
            print(f" with compute_type: {compute_type}", end="")
        print()
    _whisper_ctranslate2_model = Transcribe(
        model_path=model_name,
        device=device,
        device_index=_device_index,
        compute_type=compute_type,
        threads=1,  # Use 1 thread to avoid ctranslate2 parameter issues
        cache_directory=None,
        local_files_only=False,
        batched=False,
    )
    _whisper_ctranslate2_model_id = model_key
    return _whisper_ctranslate2_model


def download_whisper_cpp_model(model_name, dest_dir, verbose=False):
    filename = f"ggml-{model_name}.bin"
    url = f"{WHISPER_CPP_MODEL_BASE_URL}/{filename}"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    temp_path = dest_path + ".download"
    try:
        if verbose:
            print(f"Downloading whisper.cpp model: {url}")
        with urllib.request.urlopen(url) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        os.replace(temp_path, dest_path)
        return dest_path
    except Exception as exc:
        if verbose:
            print(f"Model download failed: {exc}")
        try:
            if os.path.exists(temp_path) and not getattr(globals(), "KEEP_TEMP", False):
                os.remove(temp_path)
        except Exception:
            pass
        return None


def _extract_detected_language_from_output(output_text):
    if not output_text:
        return None
    match = re.search(r"auto-detected language:\s*([a-zA-Z\-]+)", output_text, re.IGNORECASE)
    return normalize_lang_code(match.group(1)) if match else None


def _call_with_native_output_capture(func):
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except Exception:
        return func(), None

    temp_path = None
    old_stdout = None
    old_stderr = None
    temp_file = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        old_stdout = os.dup(stdout_fd)
        old_stderr = os.dup(stderr_fd)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        temp_path = temp_file.name
        temp_file.close()
        capture_fd = os.open(temp_path, os.O_WRONLY | os.O_TRUNC)
        os.dup2(capture_fd, stdout_fd)
        os.dup2(capture_fd, stderr_fd)
        os.close(capture_fd)
        result = func()
    except Exception:
        result = func()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if old_stdout is not None:
            os.dup2(old_stdout, stdout_fd)
            os.close(old_stdout)
        if old_stderr is not None:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)

    captured = None
    if temp_path and os.path.exists(temp_path):
        try:
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as handle:
                captured = handle.read()
        finally:
            try:
                if not getattr(globals(), "KEEP_TEMP", False):
                    os.remove(temp_path)
            except Exception:
                pass
    return result, captured


def _extract_detected_language_from_result(result):
    if result is None:
        return None
    if hasattr(result, "language"):
        return normalize_lang_code(getattr(result, "language"))
    if hasattr(result, "info"):
        info = getattr(result, "info")
        if hasattr(info, "language"):
            return normalize_lang_code(getattr(info, "language"))
    if isinstance(result, dict):
        for key in ("language", "lang", "detected_language"):
            if key in result:
                return normalize_lang_code(result.get(key))
        info = result.get("info") if "info" in result else None
        if isinstance(info, dict):
            for key in ("language", "lang", "detected_language"):
                if key in info:
                    return normalize_lang_code(info.get(key))
        return None
    if isinstance(result, tuple) and len(result) == 2:
        primary, info = result
        if hasattr(info, "language"):
            return normalize_lang_code(getattr(info, "language"))
        if isinstance(info, dict):
            for key in ("language", "lang", "detected_language"):
                if key in info:
                    return normalize_lang_code(info.get(key))
        result = primary
    if isinstance(result, (list, tuple)) and result:
        for seg in result:
            if hasattr(seg, "language"):
                return normalize_lang_code(getattr(seg, "language"))
            if hasattr(seg, "lang"):
                return normalize_lang_code(getattr(seg, "lang"))
            if isinstance(seg, dict):
                for key in ("language", "lang", "detected_language"):
                    if key in seg:
                        return normalize_lang_code(seg.get(key))
    return None


def _detect_language_whispercpp_cli_from_wav(wav_path, model_path, threads=1, verbose=False):
    if not wav_path or not os.path.exists(wav_path):
        return None
    config = get_whisper_cpp_config()
    bin_path = config.get("bin_path")
    if not bin_path or not model_path:
        return None

    temp_base = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as txt_fp:
            temp_base = txt_fp.name[:-4]

        cmd = [
            bin_path,
            "-m",
            model_path,
            "-f",
            wav_path,
            "-tr",
            "-otxt",
            "-of",
            temp_base,
            "-t",
            str(threads),
            "-l",
            "auto",
        ]

        if verbose:
            print(f"whispercpp CLI detect command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        combined_output = "\n".join([result.stdout or "", result.stderr or ""])
        return _extract_detected_language_from_output(combined_output)
    except Exception:
        return None
    finally:
        try:
            if temp_base and os.path.exists(f"{temp_base}.txt") and not getattr(globals(), "KEEP_TEMP", False):
                os.remove(f"{temp_base}.txt")
        except Exception:
            pass


def translate_audio_whispercpp(audio_data, samplerate=16000, input_lang=None, output_lang=None, model=None, verbose=False, timers=False, timing_stats=None):
    if verbose:
        print(f"Data type of the array: {audio_data.dtype}")
        print(f"Shape of the array: {audio_data.shape}")
        print(f"Sampling rate: {samplerate} Hz")
        print("Translating audio (whisper.cpp)...")
    timings = [] if timers else None
    t0 = time.perf_counter()
    if samplerate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
        samplerate = 16000
    add_timing(timings, "resample", t0)

    t0 = time.perf_counter()
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    add_timing(timings, "mono", t0)

    t0 = time.perf_counter()
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    add_timing(timings, "to_float32", t0)

    config = get_whisper_cpp_config()
    model_path = model or config.get("model_path")
    threads = config.get("threads", 4)

    # If model_path is just a model name (like "small") rather than a path,
    # construct the full path to the .bin file
    if model_path and not os.path.exists(model_path):
        # Check if it looks like just a model name (no path separators, no .bin extension)
        if "/" not in model_path and "\\" not in model_path and not model_path.endswith(".bin"):
            # Construct the full path: models/ggml-{name}.bin
            model_dir = os.path.dirname(config.get("model_path", "./models/ggml-medium.bin"))
            if not model_dir:
                model_dir = "./models"
            model_path = os.path.join(model_dir, f"ggml-{model_path}.bin")
            if verbose:
                print(f"Constructed model path from name: {model_path}")

    if verbose:
        print(f"whispercpp config -> model: {model_path}, threads: {threads}")

    if not model_path:
        if verbose:
            print("whispercpp model path not configured.")
        return None, None, None

    if output_lang and output_lang.lower() not in ("en", "en-us", "en-gb"):
        if verbose:
            print("whisper.cpp translate outputs English only; output_lang is ignored.")

    if get_whispercpp_force_cli():
        t0 = time.perf_counter()
        translation_text, detected_lang = translate_audio_whispercpp_cli(
            audio_data,
            samplerate=samplerate,
            input_lang=input_lang,
            output_lang=output_lang,
            model_path=model_path,
            threads=threads,
            verbose=verbose,
        )
        add_timing(timings, "transcribe", t0)

        if not translation_text and verbose:
            print("No speech detected or translation failed.")
            return None, None, None

        if is_hallucination(translation_text):
            if verbose:
                print(f"Detected hallucination, skipping: {translation_text}")
            return None, None, None

        if verbose:
            print(f"Translation ({input_lang or 'auto'} -> {output_lang or 'en'}): {translation_text}")
        if timers:
            # print(f"Timing whispercpp: {format_timing(timings)}")
            if timing_stats is not None:
                timing_stats.add(timings, prefix="whispercpp")
        return audio_data, translation_text, detected_lang

    translation_text = None
    detected_lang = None
    if input_lang and input_lang.lower() != "auto":
        detected_lang = input_lang
    temp_wav = None
    try:
        t0 = time.perf_counter()
        model_obj =  get_whispercpp_model(model_path, verbose=verbose)
        add_timing(timings, "model_load", t0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_fp:
            temp_wav = wav_fp.name
        t0 = time.perf_counter()
        sf.write(temp_wav, audio_data, samplerate, subtype="PCM_16")
        add_timing(timings, "wav_write", t0)

        transcribe_kwargs = {
            "translate": True,
            "n_threads": threads,
        }
        if input_lang:
            transcribe_kwargs["language"] = input_lang
        else:
            transcribe_kwargs["language"] = "auto"

        t0 = time.perf_counter()
        result, captured_output = _call_with_native_output_capture(
            lambda: model_obj.transcribe(temp_wav, **transcribe_kwargs)
        )
        add_timing(timings, "transcribe", t0)
        t0 = time.perf_counter()
        if detected_lang is None:
            detected_lang = _extract_detected_language_from_result(result)
        if detected_lang is None:
            detected_lang = _extract_detected_language_from_output(captured_output)
        if detected_lang is None and input_lang is None and get_whispercpp_cli_detect_lang():
            detected_lang = _detect_language_whispercpp_cli_from_wav(
                temp_wav,
                model_path=model_path,
                threads=threads,
                verbose=verbose,
            )
        add_timing(timings, "lang_detect", t0)
        if isinstance(result, str):
            translation_text = result.strip()
        elif isinstance(result, (list, tuple)):
            translation_text = " ".join(
                getattr(seg, "text", str(seg)).strip()
                for seg in result
                if str(getattr(seg, "text", seg)).strip()
            )
        else:
            translation_text = str(result).strip() if result is not None else None
    except Exception as exc:
        if verbose:
            print(f"pywhispercpp binding failed: {exc}")
            print("Attempting whispercpp CLI fallback...")
        translation_text, detected_lang_cli = translate_audio_whispercpp_cli(
            audio_data,
            samplerate=samplerate,
            input_lang=input_lang,
            output_lang=output_lang,
            model_path=model_path,
            threads=threads,
            verbose=verbose,
        )
        if detected_lang is None:
            detected_lang = detected_lang_cli
        if translation_text is None:
            return None, None, None
    finally:
        try:
            if temp_wav and os.path.exists(temp_wav) and not getattr(globals(), "KEEP_TEMP", False):
                os.remove(temp_wav)
        except Exception:
            pass

    if not translation_text and verbose:
        print("No speech detected or translation failed.")
        return None, None, None

    if is_hallucination(translation_text):
        if verbose:
            print(f"Detected hallucination, skipping: {translation_text}")
        return None, None, None

    if verbose:
        print(f"Translation ({input_lang or 'auto'} -> {output_lang or 'en'}): {translation_text}")
    if timers:
        # print(f"Timing whispercpp: {format_timing(timings)}")
        if timing_stats is not None:
            timing_stats.add(timings, prefix="whispercpp")
    return audio_data, translation_text, detected_lang


def translate_audio_faster_whisper(audio_data, samplerate=16000, input_lang=None, output_lang=None, model_size="medium", device_preference="cuda", compute_type="float16", model=None, verbose=False, timers=False, timing_stats=None):

    if verbose:
        print(f"Data type of the array: {audio_data.dtype}")
        print(f"Shape of the array: {audio_data.shape}")
        print(f"Sampling rate: {samplerate} Hz")
        print("Translating audio (faster-whisper)...")

    timings = [] if timers else None
    t0 = time.perf_counter()
    if samplerate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
        samplerate = 16000
    add_timing(timings, "resample", t0)

    t0 = time.perf_counter()
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    add_timing(timings, "mono", t0)

    t0 = time.perf_counter()
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    add_timing(timings, "to_float32", t0)

    t0 = time.perf_counter()
    model_obj = get_faster_whisper_model(model_size=model_size, device_preference=device_preference, compute_type=compute_type)
    if verbose:
        model_path = _resolve_faster_whisper_model_path(model_size, model_obj=model_obj)
        if model_path:
            print(f"faster-whisper model path: {model_path}")
        else:
            print("faster-whisper model path: unknown (not found in cache)")
    add_timing(timings, "model_load", t0)

    # Convert 'auto' to None for automatic language detection
    lang_for_whisper = None if (input_lang is None or str(input_lang).lower() == 'auto') else input_lang

    t0 = time.perf_counter()
    segments, info = model_obj.transcribe(
        audio_data,
        beam_size=1,
        task="translate",
        language=lang_for_whisper,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    add_timing(timings, "transcribe", t0)
    translation_text = " ".join([seg.text for seg in segments]).strip()
    detected_lang = None
    if input_lang and input_lang.lower() != "auto":
        detected_lang = input_lang
    else:
        detected_lang = normalize_lang_code(getattr(info, "language", None))

    if not translation_text and verbose:
        print("No speech detected or translation failed.")
        return None, None, None

    if is_hallucination(translation_text):
        if verbose:
            print(f"Detected hallucination, skipping: {translation_text}")
        return None, None, None

    if verbose:
        print(f"Translation ({input_lang or 'auto'} -> {output_lang or 'en'}): {translation_text}")
    if timers:
        # print(f"Timing faster-whisper: {format_timing(timings)}")
        if timing_stats is not None:
            timing_stats.add(timings, prefix="faster_whisper")
    return audio_data, translation_text, detected_lang


def translate_audio_whisper_ctranslate2(
    audio_data,
    samplerate=16000,
    input_lang=None,
    output_lang=None,
    model_name="small",
    device="auto",
    device_index=None,
    compute_type="default",
    model=None,
    verbose=False,
    timers=False,
    timing_stats=None,
):
    """
    Translate the given audio using a Whisper model executed via the CTranslate2 backend.

    Parameters
    ----------
    audio_data : array-like, bytes, str, or file-like
        Audio input to be processed. This may be a NumPy array containing the
        audio samples, raw bytes, a path to an audio file, or a file-like
        object, depending on how the backend is used.
    samplerate : int, optional
        Sample rate of ``audio_data`` in Hertz. Defaults to ``16000``.
    input_lang : str or None, optional
        Source language code (e.g. ``"en"``). If ``None``, language detection
        may be performed automatically by the backend.
    output_lang : str or None, optional
        Target language code for translation. If ``None``, the backend's
        default behavior is used (often English).
    model_name : str, optional
        Name or size of the Whisper-ctranslate2 model to use (e.g. ``"small"``,
        ``"medium"``). Defaults to ``"small"``.
    device : str, optional
        Device selection strategy for CTranslate2, such as ``"auto"``,
        ``"cpu"``, or ``"cuda"``. Defaults to ``"auto"``.
    device_index : int or None, optional
        Optional index of the device to use (e.g. GPU index). If ``None``,
        the backend chooses a default.
    compute_type : str, optional
        Precision or compute type to use with CTranslate2 (e.g. ``"default"``,
        ``"int8"``, ``"float16"``). Defaults to ``"default"``.
    model : object or None, optional
        An already loaded whisper-ctranslate2 model instance. If ``None``,
        the function may load or obtain a model according to configuration.
    verbose : bool, optional
        If ``True``, prints diagnostic information and progress messages.
        Defaults to ``False``.
    timers : bool, optional
        If ``True``, timing measurements are collected for different stages of
        processing. Defaults to ``False``.
    timing_stats : object or None, optional
        Optional aggregator that, if provided, receives timing information via
        its ``add`` method.

    Returns
    -------
    tuple
        A 3-tuple ``(audio_data, translation_text, detected_lang)`` where:

        * ``audio_data``: The (possibly preprocessed) audio data, typically as a
          1D NumPy array of samples at the effective sampling rate.
        * ``translation_text``: ``str`` containing the translated text.
        * ``detected_lang``: ``str`` language code that was detected or
          confirmed for the input audio.

    Raises
    ------
    Exception
        Any exception raised by underlying libraries such as ``librosa``,
        NumPy, or the whisper-ctranslate2 backend is propagated to the caller.
    """

    if verbose:
        print(f"Input audio data shape: {audio_data.shape}")
        print(f"Input samplerate: {samplerate}")
        print(f"Data type of the array: {audio_data.dtype}")

    timings = [] if timers else None
    t0 = time.perf_counter()
    if samplerate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
        samplerate = 16000
    add_timing(timings, "resample", t0)

    t0 = time.perf_counter()
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    add_timing(timings, "mono", t0)

    t0 = time.perf_counter()
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    add_timing(timings, "to_float32", t0)


    ctranslate2_config = get_whisper_ctranslate2_config()
    device_to_use = _get_with_override(device, ctranslate2_config, "device", "auto")
    device_index_to_use = _get_with_override(device_index, ctranslate2_config, "device_index")
    compute_type_to_use = _get_with_override(compute_type, ctranslate2_config, "compute_type", "default")
    if verbose:
        print(
            f"whisper-ctranslate2 config -> model: {model_name}, "
            f"device: {device_to_use}, compute_type: {compute_type_to_use}"
        )


    t0 = time.perf_counter()
    model_obj = get_whisper_ctranslate2_model(
        model_name=model_name,
        device=device_to_use,
        device_index=device_index_to_use,
        compute_type=compute_type_to_use,
        verbose=verbose,
    )
    if verbose:
        model_path = _resolve_ctranslate2_model_path(model_name, model_obj=model_obj)
        if model_path:
            print(f"whisper-ctranslate2 model path: {model_path}")
        else:
            print("whisper-ctranslate2 model path: unknown (not found in cache)")

    add_timing(timings, "model_load", t0)

    options = TranscriptionOptions(
        beam_size=5,
        best_of=5,
        patience=1.0,  # Changed from None - ctranslate2 requires a float
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=True,
        prompt_reset_on_temperature=0.5,
        temperature=0.0,
        initial_prompt=None,
        prefix=None,
        hotwords=None,
        suppress_blank=True,
        suppress_tokens=[-1],  # Changed from None - use -1 for no suppression
        word_timestamps=False,
        print_colors=False,
        prepend_punctuations="",
        append_punctuations="",
        hallucination_silence_threshold=None,
        vad_filter=False,
        vad_threshold=0.5,
        vad_min_speech_duration_ms=250,
        vad_max_speech_duration_s=None,
        vad_min_silence_duration_ms=2000,
        multilingual=True
    )

    t0 = time.perf_counter()
    translation_text = None
    try:
        # Determine task based on desired output language.
        # Whisper's "translate" task always produces English output, regardless of output_lang.
        # Use "translate" when English (or default) output is desired; otherwise fall back to "transcribe".
        normalized_output_lang = normalize_lang_code(output_lang) if output_lang else None
        if normalized_output_lang is None or normalized_output_lang in ("en", "eng", "english"):
            task = "translate"
        else:
            task = "transcribe"
            if verbose:
                print(
                    f"Warning: whisper-ctranslate2 cannot translate to non-English language "
                    f"'{output_lang}'. Falling back to transcription in the input language."
                )
        # Convert 'auto' to None for automatic language detection
        lang_for_whisper = None if (input_lang is None or str(input_lang).lower() == 'auto') else input_lang
        result = model_obj.inference(
            audio=audio_data,
            task=task,
            language=lang_for_whisper,
            verbose=verbose,
            live=False,
            options=options
        )
    except Exception as exc:
        if verbose:
            print(f"Error during whisper-ctranslate2 transcription: {exc}")
        return None, None, None
    finally:
        add_timing(timings, "transcribe", t0)


    raw_lang = result.get("language")

    detected_lang = normalize_lang_code(raw_lang) if isinstance(raw_lang, str) else None

    # Extract translation text from the transcription result
    translation_text = result.get("text")
    if not translation_text:
        segments = result.get("segments", [])
        concatenated = " ".join(
            segment.get("text", "") for segment in segments if isinstance(segment, dict) and "text" in segment
        ).strip()
        if concatenated:
            translation_text = concatenated
    if not translation_text and verbose:
        print("No speech detected or translation failed.")
        return None, None, None

    if is_hallucination(translation_text):
        if verbose:
            print(f"Detected hallucination, skipping: {translation_text}")
        return None, None, None

    if verbose:
        print(f"Translation ({input_lang or 'auto'} -> {output_lang or 'en'}): {translation_text}")
    if timers:
        # print(f"Timing whisper-ctranslate2: {format_timing(timings)}")
        if timing_stats is not None:
            timing_stats.add(timings, prefix="whisper_ctranslate2")
    return audio_data, translation_text, detected_lang


def translate_audio_whispercpp_cli(audio_data, samplerate=16000, input_lang=None, output_lang=None, model_path=None, threads=1, verbose=False):
    config = get_whisper_cpp_config()
    bin_path = config.get("bin_path")
    # bin_path and model_path are expected to be fully normalized (including ~ expansion)
    # before this function is called, e.g. via resolve_path_with_fallback.
    # Expand ~ to home directory in paths as a defensive fallback.
    # Paths are normally normalized earlier (e.g., via resolve_path_with_fallback),
    # but we re-apply expanduser here in case a raw path is passed in.
    bin_path = os.path.expanduser(bin_path)
    model_path = os.path.expanduser(model_path)

    if output_lang and output_lang.lower() not in ("en", "en-us", "en-gb"):
        if verbose:
            print("whispercpp translate outputs English only; output_lang is ignored.")

    temp_wav = None
    temp_base = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_fp:
            temp_wav = wav_fp.name
        sf.write(temp_wav, audio_data, samplerate, subtype="PCM_16")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as txt_fp:
            temp_base = txt_fp.name[:-4]

        cmd = [
            bin_path,
            "-m",
            model_path,
            "-f",
            temp_wav,
            "-tr",
            "-otxt",
            "-of",
            temp_base,
            "-t",
            str(threads),
        ]
        if input_lang:
            cmd.extend(["-l", input_lang])

        if verbose:
            print(f"whispercpp CLI command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if verbose:
                print(f"whispercpp CLI failed: {result.stderr.strip()}")
                if result.stdout:
                    print(f"whispercpp CLI stdout: {result.stdout.strip()}")
            return None, None

        detected_lang = None
        if input_lang and input_lang.lower() != "auto":
            detected_lang = input_lang
        else:
            combined_output = "\n".join([result.stdout or "", result.stderr or ""])
            detected_lang = _extract_detected_language_from_output(combined_output)

        out_txt = f"{temp_base}.txt"
        if os.path.exists(out_txt):
            with open(out_txt, "r", encoding="utf-8", errors="ignore") as handle:
                return " ".join(line.strip() for line in handle if line.strip()), detected_lang
        if verbose:
            print(f"whispercpp CLI output text not found: {out_txt}")
        return None, detected_lang
    finally:
        try:
            if temp_wav and os.path.exists(temp_wav) and not getattr(globals(), "KEEP_TEMP", False):
                os.remove(temp_wav)
        except Exception:
            pass
        try:
            if temp_base and os.path.exists(f"{temp_base}.txt") and not getattr(globals(), "KEEP_TEMP", False):
                os.remove(f"{temp_base}.txt")
        except Exception:
            pass

def get_effective_backend(backend_preference):
    """
    Determine the effective backend to use for translation based on preference and availability.
    
    Parameters
    ----------
    backend_preference : str
        Preferred backend: "whispercpp", "faster_whisper", "whisper_ctranslate2", or "auto"
    
    Returns
    -------
    str
        The name of the backend to use
    
    Raises
    ------
    RuntimeError
        If no suitable backend is available
    """
    # Normalize backend names by replacing hyphens with underscores
    normalized_preference = backend_preference.replace("-", "_") if backend_preference else "auto"
    
    if normalized_preference != "auto":
        if normalized_preference == "whispercpp_cli":
            return "whispercpp"
        return normalized_preference
    
    if _whisper_ctranslate2_available:
            return "whisper_ctranslate2"
    
    # Try backends in order of preference
    if _whispercpp_binding_available:
        return "whispercpp"
    
    # faster-whisper is always available if whisper is installed
    return "faster_whisper"

def translate_audio(audio_data, samplerate=16000, input_lang=None, output_lang=None, model="medium", device_preference="cuda", compute_type="float16", backend_preference="auto", verbose=False, timers=False, timing_stats=None):
    """
    Translate audio using the configured backend (whisper.cpp, faster-whisper, or whisper-ctranslate2).

    This function serves as a high-level interface for translating audio data. It selects
    the appropriate backend based on configuration and availability, and then delegates
    to the specific translation function for that backend.

    Parameters
    ----------
    audio_data : array-like, bytes, str, or file-like
        Audio input to be processed. This may be a NumPy array containing the
        audio samples, raw bytes, a path to an audio file, or a file-like
        object, depending on how the backend is used.
    samplerate : int, optional
        Sample rate of ``audio_data`` in Hertz. Defaults to ``16000``.                                  
    input_lang : str or None, optional
        Source language code (e.g. ``"en"``). If ``None``, language detection
        may be performed automatically by the backend.          
    output_lang : str or None, optional
        Target language code for translation. If ``None``, the backend's
        default behavior is used (often English).   
    model : str, optional
        Model size or name to use for translation (e.g. ``"small"``, ``"medium"``). Defaults to ``"medium"``.   
    device_preference : str, optional   
        Device selection strategy for CUDA usage, such as ``"auto"``, ``"cpu"``, or ``"cuda"``. Defaults to ``"cuda"``. 


    compute_type : str, optional
        Precision or compute type to use with CTranslate2 (e.g. ``"default"``, ``"int8"``, ``"float16"``). Defaults to ``"float16"``.
    backend_preference : str, optional
        Preferred backend for translation, such as ``"whispercpp"``, ``"faster_whisper"``, ``"whisper_ctranslate2"``, or ``"auto"``. Defaults to ``"auto"``.
    verbose : bool, optional    
        If ``True``, prints diagnostic information and progress messages. Defaults to ``False``.    
    timers : bool, optional     
        If ``True``, timing measurements are collected for different stages of
        processing. Defaults to ``False``.          
    timing_stats : object or None, optional                             

        
        Optional aggregator that, if provided, receives timing information via
        its ``add`` method.

            
    Returns   -------
    tuple
        A 3-tuple ``(audio_data, translation_text, detected_lang)`` where:

        * ``audio_data``: The (possibly preprocessed) audio data, typically as a
          1D NumPy array of samples at the effective sampling rate.
        * ``translation_text``: ``str`` containing the translated text.
        * ``detected_lang``: ``str`` language code that was detected or
          confirmed for the input audio.        
    Raises    ------
    Exception
        Any exception raised by underlying libraries or backends is propagated to the caller.
    """ 

    backend = get_effective_backend(backend_preference)
    if verbose:
        print(f"Selected backend for translation: {backend}")
    if backend == "whispercpp":
        return translate_audio_whispercpp(
            audio_data,
            samplerate=samplerate,
            input_lang=input_lang,
            output_lang=output_lang,
            model=model,
            verbose=verbose,
            timers=timers,
            timing_stats=timing_stats,
        )
    elif backend == "faster_whisper":
        return translate_audio_faster_whisper(
            audio_data,
            samplerate=samplerate,
            input_lang=input_lang,
            output_lang=output_lang,
            model_size=model,
            device_preference=device_preference,
            compute_type=compute_type,
            verbose=verbose,
            timers=timers,
            timing_stats=timing_stats,
        )
    elif backend == "whisper_ctranslate2":
        return translate_audio_whisper_ctranslate2(
            audio_data,
            samplerate=samplerate,
            input_lang=input_lang,
            output_lang=output_lang,
            model_name=model,
            device=device_preference,
            compute_type=compute_type,
            verbose=verbose,
            timers=timers,
            timing_stats=timing_stats,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
