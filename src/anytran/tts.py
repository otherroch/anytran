import json
import os
import sys
import subprocess
try:
    from piper.voice import PiperVoice
    PIPER_PYTHON_AVAILABLE = True
except ImportError:
    PiperVoice = None
    PIPER_PYTHON_AVAILABLE = False

try:
    from qwen_tts import Qwen3TTSModel
    QWEN_TTS_AVAILABLE = True
except ImportError:
    Qwen3TTSModel = None
    QWEN_TTS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from indextts.infer_v2 import IndexTTS2 as _IndexTTS2
    INDEXTTS_AVAILABLE = True
except Exception:
    _IndexTTS2 = None
    INDEXTTS_AVAILABLE = False

# ── Coqui TTS (coqui-tts) ──────────────────────────────────────────────────
# coqui-tts is the Python 3.12-compatible maintained fork of the original
# coqui-ai/TTS library.  Install: pip install "anytran[coqui]"
try:
    from TTS.api import TTS as _CoquiTTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    _CoquiTTS = None
    COQUI_TTS_AVAILABLE = False
except RuntimeError:
    # The original coqui-ai/TTS package (not coqui-tts) raises RuntimeError
    # on Python >= 3.12: "TTS requires python >= 3.9 and < 3.12".
    # Install the maintained fork instead: pip install coqui-tts
    _CoquiTTS = None
    COQUI_TTS_AVAILABLE = False
except Exception:
    _CoquiTTS = None
    COQUI_TTS_AVAILABLE = False

try:
    from fish_speech.inference_engine import TTSInferenceEngine as _FishTTSInferenceEngine
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue as _fish_launch_llama_queue
    from fish_speech.utils.schema import ServeReferenceAudio as _FishServeReferenceAudio
    from fish_speech.utils.schema import ServeTTSRequest as _FishServeTTSRequest
    # fish_speech/models/dac/inference.py calls pyrootutils.setup_root() at module
    # level to find the source-tree root.  When fish-speech is pip-installed the
    # .project-root marker file is absent and the call raises FileNotFoundError.
    # pip already placed everything on sys.path, so temporarily replace setup_root
    # with a no-op to let the import succeed, then restore the original.
    import pyrootutils as _pyrootutils
    _pyrootutils_orig = _pyrootutils.setup_root
    _pyrootutils.setup_root = lambda *a, **kw: None
    try:
        from fish_speech.models.dac.inference import load_model as _fish_load_decoder_model
    finally:
        _pyrootutils.setup_root = _pyrootutils_orig
    del _pyrootutils, _pyrootutils_orig
    FISH_TTS_AVAILABLE = True
except Exception:
    _FishTTSInferenceEngine = None
    _fish_load_decoder_model = None
    _fish_launch_llama_queue = None
    _FishServeReferenceAudio = None
    _FishServeTTSRequest = None
    FISH_TTS_AVAILABLE = False

import tempfile

import librosa
import numpy as np
import soundfile as sf
try:
    from gtts import gTTS
except (ImportError, AttributeError):
    gTTS = None
try:
    from playsound3 import playsound
except ImportError:
    playsound = None
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

from .voice_matcher import (
    extract_voice_features,
    select_best_piper_voice,
)


# Module-level caches used to avoid repeated voice matching and model loading.
# - _cached_matched_voice: the most recently selected voice for a given output
#   language. This is reused as long as the requested output language does not change.
# - _cached_output_lang: the language code associated with _cached_matched_voice.
#   When the requested output language differs from this value, the cached voice
#   should be considered invalid and recomputed.
# - _piper_voice_cache: a mapping from model identifiers (e.g., file paths) to
#   PiperVoice instances, so that models are loaded only once per process.
# - _custom_model_cache: a mapping from model names to Qwen3TTSModel instances,
#   so that models are loaded only once per process.
# - _fish_model_cache: a mapping from model names to TTSInferenceEngine instances,
#   so that fish-speech models are loaded only once per process.
# These caches are initialized at import time and are updated by helper functions
# in this module; they are internal implementation details and should not be
# modified directly by callers.
_cached_matched_voice = None
_cached_output_lang = None
_piper_voice_cache = {}
_custom_model_cache = {}
_fish_model_cache = {}
_indextts_model_cache = {}
_coqui_model_cache = {}

_INDEXTTS_DEFAULT_MODEL = "IndexTeam/IndexTTS-2"

# Default coqui-tts model: XTTS v2 is the multi-lingual, voice-cloning model.
# Install: pip install "anytran[coqui]"
_COQUI_DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS v2 supported language codes (ISO 639-1 / XTTS-specific).
_COQUI_XTTS_LANGUAGES = frozenset({
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
})

# Map common BCP-47 / ISO 639-1 codes to the codes expected by XTTS v2.
_COQUI_LANG_MAP = {
    "zh": "zh-cn",
    "zho": "zh-cn",
    "cmn": "zh-cn",
    "zh-tw": "zh-cn",
    "zh-hk": "zh-cn",
}

# Fish-speech model name aliases: the problem statement uses "fishaudio/s1-mini"
# but the canonical HuggingFace repo is "fishaudio/openaudio-s1-mini".
_FISH_MODEL_ALIASES = {
    "fishaudio/s1-mini": "fishaudio/openaudio-s1-mini",
}


def _normalize_fish_model_name(model_name):
    """Resolve fish-speech model aliases and return the canonical HuggingFace repo id."""
    if not model_name:
        return "fishaudio/openaudio-s1-mini"
    return _FISH_MODEL_ALIASES.get(model_name, model_name)


def _find_piper_config_path(model_path):
    if not model_path:
        return None
    candidates = []
    if model_path.endswith(".onnx"):
        candidates.append(f"{model_path}.json")  # e.g., voice.onnx.json
        base_without_ext = os.path.splitext(model_path)[0]
        candidates.append(f"{base_without_ext}.json")
    else:
        candidates.append(f"{model_path}.json")
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _resolve_piper_sample_rate(voice, config_path):
    def _try_get(obj, *attrs):
        for attr in attrs:
            if obj is None:
                return None
            obj = getattr(obj, attr, None)
        return obj

    candidates = [
        getattr(voice, "sample_rate", None),
        getattr(voice, "rate", None),
        _try_get(getattr(voice, "config", None), "sample_rate"),
        _try_get(getattr(voice, "config", None), "audio", "sample_rate"),
    ]

    for sr in candidates:
        if isinstance(sr, (int, float)) and sr > 0:
            return int(sr)

    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as cfg_fp:
                cfg = json.load(cfg_fp)
            audio = cfg.get("audio", {}) if isinstance(cfg, dict) else {}
            sr = cfg.get("sample_rate") or audio.get("sample_rate") or audio.get("sampling_rate")
            if isinstance(sr, (int, float)) and sr > 0:
                return int(sr)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    return 22050


def _ensure_gtts_available(verbose=True):
    missing = []
    if gTTS is None:
        missing.append("gTTS")
    if AudioSegment is None:
        missing.append("pydub")
    if missing:
        if verbose:
            joined = ", ".join(missing)
            print(f"[gTTS][ERROR] Missing dependencies: {joined}. Install them to use the gTTS backend.")
        return False
    return True

def ensure_piper_voice_available(voice_model, verbose=False):
    """
    Ensure a Piper voice model is available by downloading using Piper's native downloader.
    
    Parameters
    ----------
    voice_model : str
        Voice model name (e.g., "fr_FR-upmc-medium")
    verbose : bool
        Print debug information
    
    Returns
    -------
    bool
        True if voice is available or successfully downloaded, False otherwise
    """
    try:
        if not PIPER_PYTHON_AVAILABLE:
            print("[Piper][ERROR] Piper Python bindings are not installed. Please install piper-tts.")
            return False
        if verbose:
            print(f"[Voice Download] Downloading Piper voice: {voice_model} (Python API)")
        # Use piper.download_voices.download_voice method
        try:
            from piper.download_voices import download_voice
            from pathlib import Path
            models_dir = Path("./models")
            models_dir.mkdir(parents=True, exist_ok=True)
            download_voice(voice_model, models_dir)
            if verbose:
                print(f"[Voice Download] ✓ Successfully downloaded {voice_model} to {models_dir}")
            return True
        except Exception as exc:
            if verbose:
                print(f"[Voice Download] ✗ Download failed: {exc}")
            return False
    except Exception as exc:
        if verbose:
            print(f"[Voice Download] ✗ Download exception: {exc}")
        return False


def piper_tts(text, voice_model, output_wav, verbose=False):
    global _piper_voice_cache
    
    try:
        if not PIPER_PYTHON_AVAILABLE:
            print("[Piper][ERROR] Piper Python bindings are not installed. Please install piper-tts.")
            return False

        # If the voice_model is just a name, look for a matching .onnx file in the current directory
        model_path = voice_model
        if not (os.path.isfile(model_path) and model_path.endswith('.onnx')):
            candidate = os.path.abspath(f"{voice_model}.onnx")
            if os.path.isfile(candidate):
                model_path = candidate
            else:
                candidate = os.path.abspath(os.path.join("models", f"{voice_model}.onnx"))
                if os.path.isfile(candidate):
                    model_path = candidate
        config_path = _find_piper_config_path(model_path)
        cache_key = (model_path, config_path)
        if verbose:
            print(f"[Piper] Using model path: {model_path}")
            print(f"[Piper] Model file exists: {os.path.isfile(model_path)}")
            if config_path:
                print(f"[Piper] Using config path: {config_path}")
        if not os.path.isfile(model_path):
            # Attempt to auto-download the voice model before failing
            if verbose:
                print(f"[Piper] Model file missing, attempting download for voice_model='{voice_model}'")
            if ensure_piper_voice_available(voice_model, verbose=verbose):
                # After successful download, prefer the standard models directory
                downloaded_candidate = os.path.abspath(os.path.join("models", f"{voice_model}.onnx"))
                if os.path.isfile(downloaded_candidate):
                    model_path = downloaded_candidate
                    if verbose:
                        print(f"[Piper] Using downloaded model path: {model_path}")
                else:
                    if verbose:
                        print(f"[Piper] Download reported success but model file not found at: {downloaded_candidate}")
            # Final check: if still no model file, report error and fail
            if not os.path.isfile(model_path):
                print(f"[Piper][ERROR] Model file does not exist: {model_path}")
                return False

        # Refresh config/cache info if we downloaded into a new path
        config_path = _find_piper_config_path(model_path)
        cache_key = (model_path, config_path)
        if verbose and config_path:
            print(f"[Piper] Using config path: {config_path}")

        # Use Piper Python API
        try:
            voice = _piper_voice_cache.get(cache_key)
            if voice is not None:
                if verbose:
                    print(f"[Piper] Reusing cached PiperVoice instance for model: {voice_model}")
            else:
                load_kwargs = {"config_path": config_path} if config_path else {}
                voice = PiperVoice.load(model_path, **load_kwargs)
                _piper_voice_cache[cache_key] = voice
                if verbose:
                    print(f"[Piper] Loaded new PiperVoice instance for model: {voice_model}")
            if verbose:
                print(f"[Piper] Synthesizing with PiperVoice.synthesize_wav...")
            import wave
            with wave.open(output_wav, "wb") as wav_file:
                sample_rate = _resolve_piper_sample_rate(voice, config_path)
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                voice.synthesize_wav(text, wav_file)
            if verbose:
                print(f"[Piper] ✓ Synthesis successful with {voice_model} (Python API)")
            return True
        except Exception as api_exc:
            print(f"[Piper][ERROR] Piper Python API synthesis failed: {api_exc}")
            return False
    except Exception as exc:
        if verbose:
            print(f"[Piper] TTS exception: {exc}")
        return False


def _map_to_qwen_language(lang_code):
    """
    Map language code to Qwen3-TTS language name.
    
    Qwen3-TTS supports: Chinese, English, Japanese, Korean, German, French, 
    Russian, Portuguese, Spanish, Italian
    """
    if not lang_code:
        return "Auto"
    
    lang_base = lang_code.split("-")[0].split("_")[0].lower()
    
    language_map = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }
    
    return language_map.get(lang_base, "Auto")


def custom_tts(text, voice_model, output_lang, output_wav, reference_audio=None, reference_text=None, verbose=False):
    """
    Synthesize text to audio using Qwen3-TTS (CustomVoice or Base models).
    
    Parameters
    ----------
    text : str
        Text to synthesize
    voice_model : str
        Model name or path (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" or
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    output_lang : str
        Output language code (e.g., "en", "zh-CN")
    output_wav : str
        Path to output WAV file
    reference_audio : np.ndarray or None
        Reference audio for voice cloning (only used with Base model)
    reference_text : str or None
        Transcript of reference audio (improves cloning quality)
    verbose : bool
        Print debug information
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    global _custom_model_cache
    
    try:
        if not QWEN_TTS_AVAILABLE:
            print("[CustomTTS][ERROR] qwen-tts is not installed. Please install with: pip install qwen-tts")
            return False
        
        # Default to CustomVoice model if not specified or if a Piper model is specified
        if not voice_model or (voice_model and "Qwen" not in voice_model):
            if verbose and voice_model:
                print(f"[CustomTTS] Replacing non-Qwen model '{voice_model}' with default CustomVoice model")
            voice_model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        
        # Determine if this is a Base model (for voice cloning) or CustomVoice model
        is_base_model = "Base" in voice_model
        
        # Load model from cache or create new instance
        if voice_model in _custom_model_cache:
            if verbose:
                print(f"[CustomTTS] Reusing cached model: {voice_model}")
            model = _custom_model_cache[voice_model]
        else:
            if verbose:
                print(f"[CustomTTS] Loading model: {voice_model}")
            try:
                if not TORCH_AVAILABLE:
                    print("[CustomTTS][ERROR] torch is not installed. Please install with: pip install torch")
                    return False
                
                model = Qwen3TTSModel.from_pretrained(
                    voice_model,
                    device_map="auto",
                    dtype=torch.bfloat16,
                )
                _custom_model_cache[voice_model] = model
                if verbose:
                    print(f"[CustomTTS] Model loaded successfully")
            except Exception as load_exc:
                print(f"[CustomTTS][ERROR] Failed to load model: {load_exc}")
                return False
        
        # Map language code to Qwen3 language name
        language = _map_to_qwen_language(output_lang)
        if verbose:
            print(f"[CustomTTS] Language mapping: {output_lang} -> {language}")
        
        # Generate speech
        try:
            if is_base_model and reference_audio is not None:
                # Voice cloning mode
                if verbose:
                    print(f"[CustomTTS] Using voice cloning with reference audio")
                
                # Convert reference audio to proper format (numpy array with sample rate)
                # reference_audio is already a numpy array from anytran
                ref_audio_tuple = (reference_audio, 16000)  # anytran uses 16kHz
                
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio_tuple,
                    ref_text=reference_text if reference_text else None,
                )
            else:
                # CustomVoice mode - use default speaker
                if verbose:
                    print(f"[CustomTTS] Using CustomVoice synthesis")
                
                # Get supported speakers and use first one as default
                try:
                    speakers = model.get_supported_speakers()
                    default_speaker = speakers[0] if speakers else "Ryan"
                    if verbose:
                        print(f"[CustomTTS] Available speakers: {speakers}")
                        print(f"[CustomTTS] Using default speaker: {default_speaker}")
                except Exception:
                    # Fallback if get_supported_speakers is not available
                    default_speaker = "Ryan"
                    if verbose:
                        print(f"[CustomTTS] Using fallback speaker: {default_speaker}")
                
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=default_speaker,
                )
            
            # Save to WAV file
            import soundfile as sf
            sf.write(output_wav, wavs[0], sr)
            
            if verbose:
                print(f"[CustomTTS] ✓ Synthesis successful, saved to {output_wav}")
            return True
            
        except Exception as gen_exc:
            print(f"[CustomTTS][ERROR] Generation failed: {gen_exc}")
            return False
    
    except Exception as exc:
        if verbose:
            print(f"[CustomTTS] TTS exception: {exc}")
        return False


def _load_fish_engine(model_name, verbose=False):
    """
    Load a fish-speech TTSInferenceEngine for the given HuggingFace model repo.

    The function downloads the model checkpoint via ``huggingface_hub`` (or
    reuses an already-cached download) and then initialises the LLaMA language
    model queue and the VQ-GAN decoder before wrapping them in a
    ``TTSInferenceEngine`` instance.

    Returns the engine on success, or ``None`` on failure.
    """
    import traceback as _traceback

    try:
        from pathlib import Path

        if not TORCH_AVAILABLE:
            print("[FishTTS][ERROR] torch is not installed. Please install with: pip install torch")
            return None

        try:
            import torchaudio as _torchaudio  # noqa: F401
        except ImportError:
            print("[FishTTS][ERROR] torchaudio is not installed. "
                  "Please install a version matching your torch installation, e.g.: "
                  "pip install torchaudio")
            return None

        if not hasattr(_torchaudio, "list_audio_backends"):
            # list_audio_backends was removed in torchaudio 2.x nightlies and was
            # absent in very old releases.  fish-speech's ReferenceLoader calls it
            # in __init__ only to choose between "ffmpeg" and "soundfile" backends.
            # Patch a shim onto the module so the engine can always be constructed.
            import importlib.util as _ilu

            def _list_audio_backends_shim():
                backends = []
                if _ilu.find_spec("soundfile") is not None:
                    backends.append("soundfile")
                try:
                    import torchaudio.backend.ffmpeg_backend as _ffb  # noqa: F401
                    backends.append("ffmpeg")
                except ImportError:
                    pass
                return backends

            _torchaudio.list_audio_backends = _list_audio_backends_shim

        from huggingface_hub import snapshot_download

        if verbose:
            print(f"[FishTTS] Downloading/locating checkpoint: {model_name}")

        checkpoint_dir = snapshot_download(model_name)
        checkpoint_path = Path(checkpoint_dir)
        decoder_path = checkpoint_path / "codec.pth"

        if not decoder_path.exists():
            print(f"[FishTTS][ERROR] codec.pth not found in {checkpoint_dir}")
            return None

        # Pick the best available device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        precision = torch.bfloat16

        if verbose:
            print(f"[FishTTS] Loading LLaMA model on {device}...")

        llama_queue = _fish_launch_llama_queue(
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            compile=False,
        )

        if verbose:
            print(f"[FishTTS] Loading VQ-GAN decoder from {decoder_path}...")

        decoder_model = _fish_load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=decoder_path,
            device=device,
        )

        engine = _FishTTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )

        if verbose:
            print(f"[FishTTS] ✓ Engine loaded successfully")

        return engine

    except Exception as exc:
        print(f"[FishTTS][ERROR] Failed to load engine: {exc}")
        _traceback.print_exc()
        return None


def fish_tts(text, voice_model, output_wav, reference_audio=None, reference_sample_rate=16000, reference_text=None, verbose=False):
    """
    Synthesize text to audio using fish-speech (s1-mini or fish-speech-1.5).

    Parameters
    ----------
    text : str
        Text to synthesize.
    voice_model : str
        HuggingFace model repo: ``"fishaudio/s1-mini"`` (alias for
        ``"fishaudio/openaudio-s1-mini"``) or ``"fishaudio/fish-speech-1.5"``.
        When *None* or an empty string the default ``"fishaudio/openaudio-s1-mini"``
        is used.
    output_wav : str
        Path to output WAV file.
    reference_audio : np.ndarray or None
        Reference audio for voice cloning (int16 PCM or float32 in [-1, 1]).
        When provided along with *reference_text* the model performs zero-shot
        voice cloning.
    reference_sample_rate : int
        Sample rate of *reference_audio* (default: 16000 Hz).
    reference_text : str or None
        Transcript of the reference audio.  Providing an accurate transcript
        improves cloning quality.
    verbose : bool
        Print debug information.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any failure.
    """
    global _fish_model_cache

    try:
        if not FISH_TTS_AVAILABLE:
            print("[FishTTS][ERROR] fish-speech is not installed. "
                  "Please install with: pip install fish-speech")
            return False

        model_name = _normalize_fish_model_name(voice_model)

        # Load / reuse cached inference engine
        if model_name in _fish_model_cache:
            if verbose:
                print(f"[FishTTS] Reusing cached engine for model: {model_name}")
            engine = _fish_model_cache[model_name]
        else:
            if verbose:
                print(f"[FishTTS] Loading model: {model_name}")
            engine = _load_fish_engine(model_name, verbose=verbose)
            if engine is None:
                return False
            _fish_model_cache[model_name] = engine

        # Build reference list for voice cloning
        references = []
        if reference_audio is not None:
            import io
            import wave as _wave

            ref_float = reference_audio.astype(np.float32)
            # Normalise from int16 range if needed
            if ref_float.max() > 1.0 or ref_float.min() < -1.0:
                ref_float = ref_float / 32768.0

            # fish-speech expects 44.1 kHz audio
            target_sr = 44100
            if reference_sample_rate != target_sr:
                try:
                    ref_float = librosa.resample(
                        ref_float, orig_sr=reference_sample_rate, target_sr=target_sr
                    )
                except Exception as resample_exc:
                    if verbose:
                        print(f"[FishTTS] Reference audio resample failed: {resample_exc}")

            # Encode as WAV bytes
            wav_buffer = io.BytesIO()
            with _wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(target_sr)
                wf.writeframes(np.clip(ref_float * 32767, -32768, 32767).astype(np.int16).tobytes())
            wav_bytes = wav_buffer.getvalue()

            references.append(
                _FishServeReferenceAudio(
                    audio=wav_bytes,
                    text=reference_text if reference_text else "",
                )
            )

        # Build the inference request
        request = _FishServeTTSRequest(
            text=text,
            references=references,
            format="wav",
            chunk_length=200,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
        )

        if verbose:
            clone_info = " (with voice cloning)" if references else ""
            print(f"[FishTTS] Synthesizing text{clone_info}...")

        # Run inference and collect the final audio segment
        final_audio = None
        final_sr = None
        for result in engine.inference(request):
            if result.code == "final":
                final_sr, final_audio = result.audio
                break
            elif result.code == "error":
                print(f"[FishTTS][ERROR] Inference failed: {result.error}")
                return False

        if final_audio is None:
            print("[FishTTS][ERROR] No audio produced by model")
            return False

        sf.write(output_wav, final_audio, final_sr)

        if verbose:
            print(f"[FishTTS] ✓ Synthesis successful, saved to {output_wav}")
        return True

    except Exception as exc:
        print(f"[FishTTS][ERROR] TTS exception: {exc}")
        return False


def _load_indextts_engine(model_name, verbose=False):
    """
    Load an IndexTTS2 engine for the given HuggingFace model repo.

    Downloads the model checkpoint via ``huggingface_hub`` (or reuses an
    already-cached download) and initialises an ``IndexTTS2`` instance.

    Returns the engine on success, or ``None`` on failure.
    """
    import traceback as _traceback

    try:
        from pathlib import Path
        from huggingface_hub import snapshot_download

        if verbose:
            print(f"[IndexTTS] Downloading/locating checkpoint: {model_name}")

        checkpoint_dir = snapshot_download(model_name)
        cfg_path = str(Path(checkpoint_dir) / "config.yaml")

        if verbose:
            print(f"[IndexTTS] Loading model from {checkpoint_dir}...")

        engine = _IndexTTS2(
            cfg_path=cfg_path,
            model_dir=checkpoint_dir,
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )

        if verbose:
            print(f"[IndexTTS] ✓ Engine loaded successfully")

        return engine

    except Exception as exc:
        print(f"[IndexTTS][ERROR] Failed to load engine: {exc}")
        _traceback.print_exc()
        return None


def indextts_tts(text, voice_model, output_wav, reference_audio=None, reference_sample_rate=16000, verbose=False):
    """
    Synthesize text to audio using IndexTTS (IndexTeam/IndexTTS-2 or compatible).

    Parameters
    ----------
    text : str
        Text to synthesize.
    voice_model : str
        HuggingFace model repo, e.g. ``"IndexTeam/IndexTTS-2"``.
        When *None* or an empty string the default ``"IndexTeam/IndexTTS-2"`` is used.
    output_wav : str
        Path to output WAV file.
    reference_audio : np.ndarray or None
        Reference audio for voice cloning (int16 PCM or float32 in [-1, 1]).
        When provided the model clones the voice from this audio.
        When *None* a default speaker prompt must be provided by the model itself;
        most IndexTTS deployments require a prompt, so synthesis may fail without
        reference audio.
    reference_sample_rate : int
        Sample rate of *reference_audio* (default: 16000 Hz).
    verbose : bool
        Print debug information.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any failure.
    """
    global _indextts_model_cache
    extra_verbose = False
    
    try:
        if not INDEXTTS_AVAILABLE:
            print("[IndexTTS][ERROR] indextts is not installed. "
                  "Install with: GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/index-tts/index-tts.git "
                  "&& pip install 'anytran[index-tts]'")
            return False

        model_name = voice_model if voice_model else _INDEXTTS_DEFAULT_MODEL

        # Load / reuse cached engine
        if model_name in _indextts_model_cache:
            if verbose:
                print(f"[IndexTTS] Reusing cached engine for model: {model_name}")
            engine = _indextts_model_cache[model_name]
        else:
            if verbose:
                print(f"[IndexTTS] Loading model: {model_name}")
            engine = _load_indextts_engine(model_name, verbose=verbose and extra_verbose)
            if engine is None:
                return False
            _indextts_model_cache[model_name] = engine

        # IndexTTS2.infer() requires a speaker prompt WAV file path.
        # When reference_audio is provided we write it to a temp file; otherwise
        # we cannot clone a voice and synthesis will fail.
        if reference_audio is None:
            print("[IndexTTS][ERROR] reference_audio MAYBE required for IndexTTS voice synthesis. "
                  "Use --voice-match to supply a speaker prompt.")
            try:
                engine.infer(
                    spk_audio_prompt=None,
                    text=text,
                    output_path=output_wav,
                    verbose=verbose and extra_verbose,
                )
            except Exception as exc:
                print(f"[IndexTTS][ERROR] Synthesis failed: {exc}")
                return False
           
            if verbose:
                print(f"[IndexTTS] ✓ Synthesis successful, saved to {output_wav}")
            return True

        import io
        import wave as _wave

        ref_float = reference_audio.astype(np.float32)
        if ref_float.max() > 1.0 or ref_float.min() < -1.0:
            ref_float = ref_float / 32768.0

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_fp:
            ref_fp_path = ref_fp.name

        try:
            wav_buffer = io.BytesIO()
            with _wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(reference_sample_rate)
                wf.writeframes(np.clip(ref_float * 32767, -32768, 32767).astype(np.int16).tobytes())
            with open(ref_fp_path, "wb") as ref_out:
                ref_out.write(wav_buffer.getvalue())

            if verbose:
                clone_info = " (with voice cloning)"
                print(f"[IndexTTS] Synthesizing text{clone_info}...")

            engine.infer(
                spk_audio_prompt=ref_fp_path,
                text=text,
                output_path=output_wav,
                verbose=verbose and extra_verbose,
            )
        finally:
            import builtins
            if os.path.exists(ref_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                os.remove(ref_fp_path)

        if verbose:
            print(f"[IndexTTS] ✓ Synthesis successful, saved to {output_wav}")
        return True

    except Exception as exc:
        print(f"[IndexTTS][ERROR] TTS exception: {exc}")
        return False


def _map_to_coqui_language(output_lang):
    """
    Map an output language code to the code expected by coqui-tts / XTTS v2.

    XTTS v2 uses two-letter ISO 639-1 codes for most languages and ``zh-cn``
    for Mandarin Chinese.  BCP-47 region suffixes (e.g. ``en-US``) are
    stripped.  Codes not recognised by XTTS v2 fall back to ``"en"``.

    Parameters
    ----------
    output_lang : str or None
        Language tag, e.g. ``"en"``, ``"fr"``, ``"zh"``, ``"zh-CN"``.

    Returns
    -------
    str
        Language code accepted by coqui-tts XTTS v2.
    """
    if not output_lang:
        return "en"
    lang = output_lang.lower().strip()
    # Normalise region suffixes such as en-US → en, but keep zh-cn intact
    if lang in _COQUI_LANG_MAP:
        lang = _COQUI_LANG_MAP[lang]
    elif lang not in _COQUI_XTTS_LANGUAGES:
        # Strip region tag: fr-ca → fr
        base = lang.split("-")[0].split("_")[0]
        lang = _COQUI_LANG_MAP.get(base, base)
        if lang not in _COQUI_XTTS_LANGUAGES:
            lang = "en"
    return lang


def _load_coqui_engine(model_name, verbose=False):
    """
    Load a coqui-tts TTS engine for the given model name.

    The model is downloaded on first use by coqui-tts itself (stored in
    ``~/.local/share/tts``).  Subsequent calls with the same model name use
    the cached download.

    Parameters
    ----------
    model_name : str
        Coqui-tts model identifier, e.g.
        ``"tts_models/multilingual/multi-dataset/xtts_v2"``.
    verbose : bool
        Print debug information.

    Returns
    -------
    TTS or None
        Loaded ``TTS`` engine instance, or ``None`` on failure.
    """
    import traceback as _traceback

    try:
        if not COQUI_TTS_AVAILABLE:
            return None

        device = "cpu"
        if TORCH_AVAILABLE:
            import torch as _torch
            if _torch.cuda.is_available():
                device = "cuda"
            elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                device = "mps"

        if verbose:
            print(f"[CoquiTTS] Loading model '{model_name}' on {device}...")

        engine = _CoquiTTS(model_name=model_name)
        engine.to(device)

        if verbose:
            print(f"[CoquiTTS] ✓ Engine loaded successfully")

        return engine

    except Exception as exc:
        print(f"[CoquiTTS][ERROR] Failed to load engine: {exc}")
        _traceback.print_exc()
        return None


def coqui_tts(text, voice_model, output_lang, output_wav,
              reference_audio=None, reference_sample_rate=16000, verbose=False):
    """
    Synthesize text to audio using coqui-tts (Python 3.12-compatible fork).

    coqui-tts is the actively maintained fork of the original coqui-ai/TTS
    library that supports Python 3.9–3.14.  Install: ``pip install "anytran[coqui]"``.

    The default model is XTTS v2
    (``tts_models/multilingual/multi-dataset/xtts_v2``), which supports 17
    languages and zero-shot voice cloning from a short audio reference.

    Parameters
    ----------
    text : str
        Text to synthesize.
    voice_model : str or None
        Coqui-tts model identifier.  When *None* or empty the default XTTS v2
        model is used.
    output_lang : str or None
        BCP-47 / ISO 639-1 language code for the synthesized speech.
    output_wav : str
        Path to the output WAV file.
    reference_audio : np.ndarray or None
        Reference audio for zero-shot voice cloning (int16 PCM or float32 in
        ``[-1, 1]``).  When *None* the model synthesizes using its default
        speaker (the engine chooses the speaker automatically).
    reference_sample_rate : int
        Sample rate of *reference_audio* (default: 16000 Hz).
    verbose : bool
        Print debug information.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any failure.
    """
    global _coqui_model_cache

    try:
        if not COQUI_TTS_AVAILABLE:
            print("[CoquiTTS][ERROR] coqui-tts is not installed. "
                  "Install with: pip install 'anytran[coqui]'")
            return False

        model_name = voice_model if voice_model else _COQUI_DEFAULT_MODEL

        # Load / reuse cached engine
        if model_name in _coqui_model_cache:
            if verbose:
                print(f"[CoquiTTS] Reusing cached engine for model: {model_name}")
            engine = _coqui_model_cache[model_name]
        else:
            if verbose:
                print(f"[CoquiTTS] Loading model: {model_name}")
            engine = _load_coqui_engine(model_name, verbose=verbose)
            if engine is None:
                return False
            _coqui_model_cache[model_name] = engine

        lang = _map_to_coqui_language(output_lang)

        if reference_audio is not None:
            import io
            import wave as _wave

            ref_float = reference_audio.astype(np.float32)
            if ref_float.max() > 1.0 or ref_float.min() < -1.0:
                ref_float = ref_float / 32768.0

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_fp:
                ref_fp_path = ref_fp.name

            try:
                wav_buffer = io.BytesIO()
                with _wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(reference_sample_rate)
                    wf.writeframes(
                        np.clip(ref_float * 32767, -32768, 32767).astype(np.int16).tobytes()
                    )
                with open(ref_fp_path, "wb") as ref_out:
                    ref_out.write(wav_buffer.getvalue())

                if verbose:
                    print(f"[CoquiTTS] Synthesizing with voice cloning (lang={lang})...")

                if getattr(engine, "is_multi_lingual", False):
                    engine.tts_to_file(
                        text=text,
                        speaker_wav=ref_fp_path,
                        language=lang,
                        file_path=output_wav,
                    )
                else:
                    engine.tts_to_file(
                        text=text,
                        speaker_wav=ref_fp_path,
                        file_path=output_wav,
                    )
            finally:
                import builtins
                if os.path.exists(ref_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(ref_fp_path)

        else:
            if verbose:
                print(f"[CoquiTTS] Synthesizing (lang={lang})...")
            # For multi-speaker models (e.g. XTTS v2), a speaker must always be
            # provided.  Pick the first available built-in speaker when the caller
            # has not supplied a reference audio clip for voice cloning.
            speakers = getattr(engine, "speakers", None)
            default_speaker = speakers[0] if speakers else None
            if default_speaker and verbose:
                print(f"[CoquiTTS] Using default speaker: {default_speaker}")
            if getattr(engine, "is_multi_lingual", False):
                kw = {"text": text, "language": lang, "file_path": output_wav}
                if default_speaker is not None:
                    kw["speaker"] = default_speaker
                engine.tts_to_file(**kw)
            else:
                kw = {"text": text, "file_path": output_wav}
                if default_speaker is not None:
                    kw["speaker"] = default_speaker
                engine.tts_to_file(**kw)

        if verbose:
            print(f"[CoquiTTS] ✓ Synthesis successful, saved to {output_wav}")
        return True

    except Exception as exc:
        print(f"[CoquiTTS][ERROR] TTS exception: {exc}")
        return False


def play_output(translated_text, lang="en", play_audio=True, wav_file=None, rate=16000, voice_backend="gtts", voice_model=None):
    use_piper = voice_backend == "piper"
    piper_voice = voice_model

    # print(f"[TRACE] play_output called with use_piper={use_piper}, piper_voice={piper_voice}, lang={lang}, play_audio={play_audio}, wav_file={wav_file}, rate={rate}")
    if not translated_text:
        return

    try:
        if use_piper and piper_voice:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name

            try:
                if piper_tts(translated_text, piper_voice, tts_fp_path, verbose=False):
                    if play_audio:
                        if sys.platform == "win32":
                            subprocess.Popen(
                                ["ffplay", "-nodisp", "-autoexit", tts_fp_path],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                            )
                        else:
                            subprocess.Popen(
                                ["ffplay", "-nodisp", "-autoexit", tts_fp_path],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                    if wav_file:
                        audio_data, sr = sf.read(tts_fp_path)
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                        wav_file.writeframes((audio_data * 32768).astype(np.int16).tobytes())
                else:
                    use_piper = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        if not use_piper:
            if not _ensure_gtts_available():
                return
            tts_lang = lang.split("-")[0]
            tts = gTTS(text=translated_text, lang=tts_lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_fp:
                tts.save(tts_fp.name)
                tts_fp_path = tts_fp.name
            try:
                if play_audio:
                    if playsound is None:
                        print("[PlaySound][WARN] playsound3 is not installed; skipping playback.")
                    else:
                        playsound(tts_fp_path)
                if wav_file:
                    tts_audio = AudioSegment.from_mp3(tts_fp_path)
                    wav_data = tts_audio.set_frame_rate(rate).set_channels(1).set_sample_width(2)
                    wav_file.writeframes(wav_data.raw_data)
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)
    except Exception as exc:
        print(f"TTS playback failed: {exc}")


def synthesize_tts_pcm(translated_text, rate, output_lang, voice_backend="gtts", voice_model=None, verbose=False):
    
    global _cached_matched_voice
    global _cached_output_lang 
    
    
    use_piper = voice_backend == "piper"
    use_custom = voice_backend == "custom"
    use_fish = voice_backend == "fish"
    use_indextts = voice_backend == "indextts"
    use_coqui = voice_backend == "coqui"
    piper_voice = voice_model
    custom_model = voice_model
    fish_model = voice_model
    indextts_model = voice_model
    coqui_model = voice_model
    lang_base = (output_lang or "en").split("-")[0].split("_")[0].lower()
    
    if not translated_text:
        return None

    tts_pcm = None
    try:
        # Fish TTS backend (fish-speech)
        if use_fish:
            if verbose:
                print(f"[TTS] Using Fish (fish-speech) backend")

            # Replace non-fish model names (e.g. default Piper voice) with the
            # default fish-speech model
            if not fish_model or "/" not in fish_model:
                if verbose and fish_model:
                    print(f"[TTS] Replacing non-fish model '{fish_model}' with default fish-speech model")
                fish_model = "fishaudio/openaudio-s1-mini"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                fish_success = fish_tts(
                    translated_text,
                    fish_model,
                    tts_fp_path,
                    verbose=verbose,
                )
                if fish_success:
                    if verbose:
                        print(f"[TTS] Fish result: SUCCESS (model={fish_model})")
                    audio_data, sr = sf.read(tts_fp_path)
                    if not audio_data.size:
                        if verbose:
                            print(f"[TTS] Fish result: no data")
                        use_fish = False
                    else:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                            if not audio_data.size:
                                if verbose:
                                    print(f"[TTS] librosa resample result: no data")
                                use_fish = False
                        if use_fish and audio_data.size:
                            tts_pcm = (audio_data * 32768).astype(np.int16)
                            if verbose:
                                print(f"[TTS] Fish synthesis complete: {len(tts_pcm)} samples")
                            return tts_pcm
                else:
                    if verbose:
                        print(f"[TTS] Fish TTS failed, falling back to gTTS")
                    use_fish = False
            except Exception as fish_exc:
                if verbose:
                    print(f"[TTS] Fish synthesis failed: {fish_exc}")
                use_fish = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        # IndexTTS backend (IndexTeam/IndexTTS-2)
        if use_indextts:
            if verbose:
                print(f"[TTS] Using IndexTTS backend")

            if not indextts_model or "/" not in indextts_model:
                if verbose and indextts_model:
                    print(f"[TTS] Replacing non-IndexTTS model '{indextts_model}' with default IndexTTS model")
                indextts_model = _INDEXTTS_DEFAULT_MODEL

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                indextts_success = indextts_tts(
                    translated_text,
                    indextts_model,
                    tts_fp_path,
                    verbose=verbose,
                )
                if indextts_success:
                    if verbose:
                        print(f"[TTS] IndexTTS result: SUCCESS (model={indextts_model})")
                    audio_data, sr = sf.read(tts_fp_path)
                    if not audio_data.size:
                        if verbose:
                            print(f"[TTS] IndexTTS result: no data")
                        use_indextts = False
                    else:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                            if not audio_data.size:
                                if verbose:
                                    print(f"[TTS] librosa resample result: no data")
                                use_indextts = False
                        if use_indextts and audio_data.size:
                            tts_pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                            if verbose:
                                print(f"[TTS] IndexTTS synthesis complete: {len(tts_pcm)} samples")
                            return tts_pcm
                else:
                    if verbose:
                        print(f"[TTS] IndexTTS failed, falling back to gTTS")
                    use_indextts = False
            except Exception as indextts_exc:
                if verbose:
                    print(f"[TTS] IndexTTS synthesis failed: {indextts_exc}")
                use_indextts = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        # Coqui TTS backend (coqui-tts / XTTS v2)
        if use_coqui:
            if verbose:
                print(f"[TTS] Using Coqui (coqui-tts) backend")

            # Replace non-coqui model identifiers with the default XTTS v2 model
            if not coqui_model or not coqui_model.startswith("tts_models/"):
                if verbose and coqui_model:
                    print(f"[TTS] Replacing non-coqui model '{coqui_model}' with default coqui model")
                coqui_model = _COQUI_DEFAULT_MODEL

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                coqui_success = coqui_tts(
                    translated_text,
                    coqui_model,
                    output_lang,
                    tts_fp_path,
                    verbose=verbose,
                )
                if coqui_success:
                    if verbose:
                        print(f"[TTS] Coqui result: SUCCESS (model={coqui_model})")
                    audio_data, sr = sf.read(tts_fp_path)
                    if not audio_data.size:
                        if verbose:
                            print(f"[TTS] Coqui result: no data")
                        use_coqui = False
                    else:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                            if not audio_data.size:
                                if verbose:
                                    print(f"[TTS] librosa resample result: no data")
                                use_coqui = False
                        if use_coqui and audio_data.size:
                            tts_pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                            if verbose:
                                print(f"[TTS] Coqui synthesis complete: {len(tts_pcm)} samples")
                            return tts_pcm
                else:
                    if verbose:
                        print(f"[TTS] Coqui TTS failed, falling back to gTTS")
                    use_coqui = False
            except Exception as coqui_exc:
                if verbose:
                    print(f"[TTS] Coqui synthesis failed: {coqui_exc}")
                use_coqui = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        # Custom TTS backend (Qwen3-TTS)
        if use_custom:
            if verbose:
                print(f"[TTS] Using Custom (Qwen3-TTS) backend")
            
            # Default to CustomVoice model if not specified or if a Piper model is specified
            # (e.g., when using default --voice-model en_US-lessac-medium with --voice-backend custom)
            if not custom_model or (custom_model and "Qwen" not in custom_model):
                if verbose and custom_model:
                    print(f"[TTS] Replacing non-Qwen model '{custom_model}' with default CustomVoice model")
                custom_model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                custom_success = custom_tts(
                    translated_text, 
                    custom_model, 
                    output_lang,
                    tts_fp_path, 
                    verbose=verbose
                )
                if custom_success:
                    if verbose:
                        print(f"[TTS] Custom result: SUCCESS (model={custom_model})")
                    audio_data, sr = sf.read(tts_fp_path)
                    if not audio_data.size:
                        if verbose:
                            print(f"[TTS] Custom result: no data")
                        use_custom = False
                    else:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                            if not audio_data.size:
                                if verbose:
                                    print(f"[TTS] librosa resample result: no data")
                                use_custom = False
                        if use_custom and audio_data.size:
                            tts_pcm = (audio_data * 32768).astype(np.int16)
                            if verbose: 
                                print(f"[TTS] Custom synthesis complete: {len(tts_pcm)} samples")
                            return tts_pcm
                else:
                    if verbose:
                        print(f"[TTS] Custom TTS failed, falling back to gTTS")
                    use_custom = False
            except Exception as custom_exc:
                if verbose:
                    print(f"[TTS] Custom synthesis failed: {custom_exc}")
                use_custom = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)
        
        if use_piper and (piper_voice is None or piper_voice == "en_US-lessac-medium"):
            if (
                _cached_matched_voice is not None
                and output_lang is not None
                and _cached_output_lang is not None
                and output_lang == _cached_output_lang
            ):
                if verbose:
                    print(f"[TTS] Using previously matched voice: {_cached_matched_voice}")
                piper_voice = _cached_matched_voice
            elif lang_base == "en":
                # For English, default to a commonly available voice
                piper_voice = "en_US-lessac-medium"
                _cached_matched_voice = piper_voice
                _cached_output_lang = output_lang
                if verbose:
                    print(f"[TTS] Defaulting to English Piper voice: {piper_voice}")    
            else:
                neutral_features = {
                    "mean_pitch": 150.0,
                    "gender": "male",
                    "pitch_std": 0.0,
                    "zcr": 0.1,
                    "brightness": 2000.0,

                }
                lang_voice = select_best_piper_voice(neutral_features, output_lang, verbose=verbose)
                if lang_voice:
                    if verbose:
                        print(f"[TTS] Auto-selected {output_lang} Piper voice: {lang_voice} (language-aware selection)")
                    piper_voice = lang_voice
                    _cached_matched_voice = piper_voice
                    _cached_output_lang = output_lang   
                else:
                    if verbose:
                        print(f"[TTS] No suitable Piper voice found for {output_lang}, will attempt gTTS fallback")     
        
        # If Piper is requested but language has limited voices, prefer gTTS instead
        if use_piper and piper_voice is None:
            if verbose:
                print(f"[TTS] {lang_base.upper()} has limited Piper voices, preferring gTTS")
            use_piper = False
        
        if use_piper and piper_voice:
            if verbose:
                print(f"[TTS] Using Piper with voice: {piper_voice}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                piper_success = piper_tts(translated_text, piper_voice, tts_fp_path, verbose=verbose)
                if piper_success:
                    if verbose:
                        print(f"[TTS] Piper result: SUCCESS (piper_voice={piper_voice})")
                    audio_data, sr = sf.read(tts_fp_path)
                    if not audio_data.size:
                        if verbose:
                            print(f"[TTS] Piper result: no data")
                        # Treat empty output as a Piper failure and fall back to gTTS
                        use_piper = False
                    else:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                            if not audio_data.size:
                                if verbose:
                                    print(f"[TTS] librosa resample result: no data")
                                # Resampling produced no data; fall back to gTTS
                                use_piper = False
                        if use_piper and audio_data.size:
                            tts_pcm = (audio_data * 32768).astype(np.int16)
                            if verbose: 
                                print(f"[TTS] Piper synthesis complete: {len(tts_pcm)} samples")
                            return tts_pcm
                else:
                    if verbose:
                        print(f"[TTS] Piper failed, falling back to gTTS")
                    use_piper = False
            except Exception as piper_exc:
                if verbose:
                    print(f"[TTS] Piper synthesis failed: {piper_exc}")
                # Ensure Piper failures trigger gTTS fallback
                use_piper = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        if not use_piper and not use_custom and not use_fish and not use_indextts and not use_coqui:
            if not _ensure_gtts_available(verbose=verbose):
                return None
            tts_lang = lang_base
            if verbose:
                print(f"[TTS] Using gTTS with language: {tts_lang}")
            try:
                tts = gTTS(text=translated_text, lang=tts_lang)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_fp:
                    tts.save(tts_fp.name)
                    tts_fp_path = tts_fp.name
                try:
                    tts_audio = AudioSegment.from_mp3(tts_fp_path)
                    tts_audio = tts_audio.set_frame_rate(rate).set_channels(1).set_sample_width(2)
                    tts_pcm = np.frombuffer(tts_audio.raw_data, dtype=np.int16)
                    if verbose:
                        print(f"[TTS] gTTS synthesis complete: {len(tts_pcm)} samples")
                finally:
                    import builtins
                    if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                        os.remove(tts_fp_path)
            except Exception as gtts_exc:
                if verbose:
                    print(f"[TTS] gTTS failed: {gtts_exc}")
                    print(f"[TTS] Falling back to English gTTS")
                # Fallback to English if gTTS fails
                if not _ensure_gtts_available(verbose=verbose):
                    return None
                try:
                    tts = gTTS(text=translated_text, lang="en")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_fp:
                        tts.save(tts_fp.name)
                        tts_fp_path = tts_fp.name
                    try:
                        tts_audio = AudioSegment.from_mp3(tts_fp_path)
                        tts_audio = tts_audio.set_frame_rate(rate).set_channels(1).set_sample_width(2)
                        tts_pcm = np.frombuffer(tts_audio.raw_data, dtype=np.int16)
                    finally:
                        import builtins
                        if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                            os.remove(tts_fp_path)
                except Exception as fallback_exc:
                    if verbose:
                        print(f"[TTS] All TTS methods failed: {fallback_exc}")
                    return None
    except Exception as exc:
        if verbose:
            print(f"TTS synthesis for output file failed: {exc}")
            tts_pcm = None
    return tts_pcm
        
def synthesize_tts_pcm_with_cloning(
        translated_text,
        rate,
        output_lang,
        reference_audio=None,
        reference_sample_rate=16000,
        reference_text=None,
        voice_backend="gtts",
        voice_model=None,
        voice_match=False,
        verbose=False,
    ) :
    """
    Synthesize TTS with optional voice matching.
    
    Parameters
    ----------
    translated_text : str
        Text to synthesize
    rate : int
        Target sample rate
    output_lang : str
        Output language code
    reference_audio : np.ndarray or None
        Reference audio for voice matching
    reference_sample_rate : int
        Sample rate of reference audio
    reference_text : str or None
        Transcript of reference audio (improves voice cloning quality for custom backend)
    voice_backend : str
        TTS backend to use: ``"piper"``, ``"gtts"``, ``"custom"``, ``"fish"``, ``"indextts"``, or ``"coqui"`` (default: ``"gtts"``)
    voice_model : str or None
        Voice model name for TTS (used as the Piper voice when ``voice_backend`` is ``"piper"``,
        the Qwen3-TTS model when ``voice_backend`` is ``"custom"``, the fish-speech model
        when ``voice_backend`` is ``"fish"``, the IndexTTS model when ``voice_backend`` is ``"indextts"``,
        or the coqui-tts model when ``voice_backend`` is ``"coqui"``)
    voice_match : bool
        Auto-select Piper voice based on input voice features (for piper backend),
        use voice cloning with reference audio (for custom backend),
        perform zero-shot voice cloning (for fish backend),
        clone speaker voice from reference audio (for indextts backend),
        or perform zero-shot voice cloning with coqui-tts (for coqui backend)
    verbose : bool
        Print debug information
    
    Returns
    -------
    np.ndarray or None
        PCM audio data as int16 array
    
    Notes
    -----
    When ``voice_match`` is enabled, this function performs voice matching
    only once per process lifetime when reference audio is provided. The
    matched voice is stored in a module-level cache and reused for later
    calls to avoid repeated analysis.
    
    For custom backend with voice_match, the Base model is used with reference
    audio for voice cloning.

    For fish backend with voice_match, zero-shot voice cloning is performed
    using the reference audio as the speaker prompt.

    For indextts backend with voice_match, voice cloning is performed using
    the reference audio as the speaker prompt.

    For coqui backend with voice_match, zero-shot voice cloning is performed
    using the reference audio as the speaker prompt (XTTS v2).
    """ 
  
    
    use_piper = voice_backend == "piper"
    use_custom = voice_backend == "custom"
    use_fish = voice_backend == "fish"
    use_indextts = voice_backend == "indextts"
    use_coqui = voice_backend == "coqui"
    piper_voice = voice_model
    custom_model = voice_model
    fish_model = voice_model
    indextts_model = voice_model
    coqui_model = voice_model

    if not translated_text:
        return None
    
    try:
        global _cached_matched_voice
        global _cached_output_lang

        # Fish backend with voice matching uses zero-shot voice cloning
        if use_fish and voice_match and reference_audio is not None:
            if verbose:
                print("==========================================")
                print("FISH VOICE CLONING (--voice-match)")
                print("==========================================")
                print(f"Using reference audio for zero-shot voice cloning with fish-speech")

            # Replace non-fish model names with the default fish-speech model
            if not fish_model or "/" not in fish_model:
                if verbose and fish_model:
                    print(f"[TTS] Replacing non-fish model '{fish_model}' with default fish-speech model")
                fish_model = "fishaudio/openaudio-s1-mini"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                fish_success = fish_tts(
                    translated_text,
                    fish_model,
                    tts_fp_path,
                    reference_audio=reference_audio,
                    reference_sample_rate=reference_sample_rate,
                    reference_text=reference_text,
                    verbose=verbose,
                )

                if fish_success:
                    if verbose:
                        print(f"[TTS] Fish voice cloning result: SUCCESS")
                    audio_data, sr = sf.read(tts_fp_path)
                    if audio_data.size:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                        if audio_data.size:
                            tts_pcm = (audio_data * 32768).astype(np.int16)
                            if verbose:
                                print(f"[TTS] Fish voice cloning complete: {len(tts_pcm)} samples")
                                print("==========================================")
                            return tts_pcm

                if verbose:
                    print(f"[TTS] Fish voice cloning failed, falling back to standard synthesis")
                    print("==========================================")
                use_fish = False

            except Exception as fish_exc:
                if verbose:
                    print(f"[TTS] Fish voice cloning failed: {fish_exc}")
                    print("==========================================")
                use_fish = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        # IndexTTS backend with voice matching uses voice cloning with reference audio
        if use_indextts and voice_match and reference_audio is not None:
            if verbose:
                print("==========================================")
                print("INDEXTTS VOICE CLONING (--voice-match)")
                print("==========================================")
                print(f"Using reference audio for voice cloning with IndexTTS")

            if not indextts_model or "/" not in indextts_model:
                if verbose and indextts_model:
                    print(f"[TTS] Replacing non-IndexTTS model '{indextts_model}' with default IndexTTS model")
                indextts_model = _INDEXTTS_DEFAULT_MODEL

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                indextts_success = indextts_tts(
                    translated_text,
                    indextts_model,
                    tts_fp_path,
                    reference_audio=reference_audio,
                    reference_sample_rate=reference_sample_rate,
                    verbose=verbose,
                )

                if indextts_success:
                    if verbose:
                        print(f"[TTS] IndexTTS voice cloning result: SUCCESS")
                    audio_data, sr = sf.read(tts_fp_path)
                    if audio_data.size:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                        if audio_data.size:
                            audio_data = np.clip(audio_data, -1.0, 1.0)
                            tts_pcm = (audio_data * 32767).astype(np.int16)
                            if verbose:
                                print(f"[TTS] IndexTTS voice cloning complete: {len(tts_pcm)} samples")
                                print("==========================================")
                            return tts_pcm

                if verbose:
                    print(f"[TTS] IndexTTS voice cloning failed, falling back to standard synthesis")
                    print("==========================================")
                use_indextts = False

            except Exception as indextts_exc:
                if verbose:
                    print(f"[TTS] IndexTTS voice cloning failed: {indextts_exc}")
                    print("==========================================")
                use_indextts = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        # Coqui backend with voice matching uses XTTS v2 zero-shot voice cloning
        if use_coqui and voice_match and reference_audio is not None:
            if verbose:
                print("==========================================")
                print("COQUI VOICE CLONING (--voice-match)")
                print("==========================================")
                print(f"Using reference audio for zero-shot voice cloning with coqui-tts XTTS v2")

            if not coqui_model or not coqui_model.startswith("tts_models/"):
                if verbose and coqui_model:
                    print(f"[TTS] Replacing non-coqui model '{coqui_model}' with default coqui model")
                coqui_model = _COQUI_DEFAULT_MODEL

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                coqui_success = coqui_tts(
                    translated_text,
                    coqui_model,
                    output_lang,
                    tts_fp_path,
                    reference_audio=reference_audio,
                    reference_sample_rate=reference_sample_rate,
                    verbose=verbose,
                )

                if coqui_success:
                    if verbose:
                        print(f"[TTS] Coqui voice cloning result: SUCCESS")
                    audio_data, sr = sf.read(tts_fp_path)
                    if audio_data.size:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                        if audio_data.size:
                            tts_pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                            if verbose:
                                print(f"[TTS] Coqui voice cloning complete: {len(tts_pcm)} samples")
                                print("==========================================")
                            return tts_pcm

                if verbose:
                    print(f"[TTS] Coqui voice cloning failed, falling back to standard synthesis")
                    print("==========================================")
                use_coqui = False

            except Exception as coqui_exc:
                if verbose:
                    print(f"[TTS] Coqui voice cloning failed: {coqui_exc}")
                    print("==========================================")
                use_coqui = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)

        # Custom backend with voice matching uses Base model for voice cloning
        if use_custom and voice_match and reference_audio is not None:
            if verbose:
                print("==========================================")
                print("CUSTOM VOICE CLONING (--voice-match)")
                print("==========================================")
                print(f"Using reference audio for voice cloning with Qwen3-TTS Base model")
            
            # Default to Base model for voice cloning
            # Also replace Piper models (e.g., en_US-lessac-medium) with Qwen3-TTS Base
            if not custom_model or (custom_model and "Qwen" not in custom_model):
                if verbose and custom_model:
                    print(f"[TTS] Replacing non-Qwen model '{custom_model}' with default Base model")
                custom_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            elif "CustomVoice" in custom_model:
                # If CustomVoice was specified, switch to Base for cloning
                custom_model = custom_model.replace("CustomVoice", "Base")
                if verbose:
                    print(f"Switching to Base model for voice cloning: {custom_model}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                # Resample reference audio to 16kHz if needed
                ref_audio = reference_audio
                if reference_sample_rate != 16000:
                    ref_audio = librosa.resample(
                        reference_audio.astype(np.float32) / 32768.0,
                        orig_sr=reference_sample_rate,
                        target_sr=16000
                    )
                else:
                    ref_audio = reference_audio.astype(np.float32) / 32768.0
                
                custom_success = custom_tts(
                    translated_text,
                    custom_model,
                    output_lang,
                    tts_fp_path,
                    reference_audio=ref_audio,
                    reference_text=reference_text,
                    verbose=verbose
                )
                
                if custom_success:
                    if verbose:
                        print(f"[TTS] Custom voice cloning result: SUCCESS")
                    audio_data, sr = sf.read(tts_fp_path)
                    if audio_data.size:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                        if audio_data.size:
                            tts_pcm = (audio_data * 32768).astype(np.int16)
                            if verbose:
                                print(f"[TTS] Custom voice cloning complete: {len(tts_pcm)} samples")
                                print("==========================================")
                            return tts_pcm
                
                if verbose:
                    print(f"[TTS] Custom voice cloning failed, falling back to standard synthesis")
                    print("==========================================")
                use_custom = False
                
            except Exception as custom_exc:
                if verbose:
                    print(f"[TTS] Custom voice cloning failed: {custom_exc}")
                    print("==========================================")
                use_custom = False
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)
        
        # Only apply if user didn't explicitly specify a non-default voice
        # Default voice is "en_US-lessac-medium"
        explicit_voice_provided = bool(piper_voice) and piper_voice != "en_US-lessac-medium"
        
        if (
            output_lang
            and _cached_matched_voice is not None
            and _cached_output_lang is not None
            and output_lang != _cached_output_lang
        ):
            if verbose:
                print(f"[TTS] Output language changed from {_cached_output_lang} to {output_lang}, resetting cached matched voice")
            _cached_matched_voice = None
            _cached_output_lang = output_lang    
          
        cached_auto_match_applicable = voice_match and reference_audio is not None and not explicit_voice_provided
        if cached_auto_match_applicable and _cached_matched_voice is None:
            if verbose:
                print("==========================================")
                print("AUTO-VOICE MATCHING (--voice-match)")
                print("==========================================")
                print(f"Analyzing input voice characteristics...")
            
            # Extract voice features
            features = extract_voice_features(reference_audio, reference_sample_rate, verbose=verbose)
            
            # Select best matching Piper voice
            if verbose:
                print(f"Selecting best {output_lang} voice for detected characteristics...")
            matched_voice = select_best_piper_voice(features, output_lang, verbose=verbose)
            
            if matched_voice:
                if verbose:
                    print(f"✓ Voice matching successful: {matched_voice}")
                    print("==========================================")
                piper_voice = matched_voice
                _cached_matched_voice = matched_voice
                _cached_output_lang = output_lang
                use_piper = True
            else:
                if verbose:
                    print("✗ No suitable voice found, using default")
                    print("==========================================")
        elif voice_match and reference_audio is not None and explicit_voice_provided:
            if verbose:
                print("Explicit voice provided (--voice-model), skipping --voice-match")
        elif voice_match and reference_audio is None:
            if verbose:
                print("⚠ Auto-match-voice requested but no reference audio available")
        
        if _cached_matched_voice is not None:
            if verbose:
                print(f"Using previously matched voice: {_cached_matched_voice} for language {_cached_output_lang}")
            piper_voice = _cached_matched_voice
            use_piper = True

        # Language-aware automatic voice selection:
        # When Piper is requested with the default English voice but a non-English
        # output language, automatically select the best available Piper voice for
        # that language so the synthesized speech sounds natural.
        if use_piper and not explicit_voice_provided and _cached_matched_voice is None:
            lang_base = (output_lang or "en").split("-")[0].split("_")[0].lower()
            if lang_base == "en":
                    # For English, default to a commonly available voice
                piper_voice = "en_US-lessac-medium"
                if verbose: 
                    print(f"[TTS] Defaulting to English Piper voice: {piper_voice}")    
            else:
                neutral_features = {
                    "mean_pitch": 150.0,
                    "gender": "male",
                    "pitch_std": 0.0,
                    "zcr": 0.1,
                    "brightness": 2000.0,
                }
                lang_voice = select_best_piper_voice(neutral_features, output_lang, verbose=verbose)
                if lang_voice:
                    if verbose:
                        print(f"[TTS] Auto-selected {output_lang} Piper voice: {lang_voice} (language-aware selection)")
                    piper_voice = lang_voice

        # Standard TTS synthesis (Coqui, IndexTTS, Fish, Piper, Custom, or gTTS)
        if use_coqui:
            backend = "coqui"
            model = coqui_model if (coqui_model and coqui_model.startswith("tts_models/")) else _COQUI_DEFAULT_MODEL
        elif use_indextts:
            backend = "indextts"
            model = indextts_model if (indextts_model and "/" in indextts_model) else _INDEXTTS_DEFAULT_MODEL
        elif use_fish:
            backend = "fish"
            model = fish_model if (fish_model and "/" in fish_model) else "fishaudio/openaudio-s1-mini"
        elif use_custom:
            backend = "custom"
            model = custom_model if custom_model else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        elif use_piper:
            backend = "piper"
            model = piper_voice
        else:
            backend = "gtts"
            model = None
        
        tts_pcm = synthesize_tts_pcm(
            translated_text, rate, output_lang,
            voice_backend=backend,
            voice_model=model,
            verbose=verbose,
        )
        return tts_pcm
        
    except Exception as exc:
        if verbose:
            print(f"TTS synthesis with voice features failed: {exc}")
        return None
