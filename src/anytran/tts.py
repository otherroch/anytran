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

# CosyVoice import detection
try:
    from cosyvoice.cli.cosyvoice import AutoModel as CosyVoice
    COSYVOICE_AVAILABLE = True
except ImportError:
    CosyVoice = None
    COSYVOICE_AVAILABLE = False

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
# - _cosyvoice_model_cache: a mapping from model identifiers to CosyVoice instances,
#   so that CosyVoice models are loaded only once per process.
# - _custom_model_cache: a mapping from model names to Qwen3TTSModel instances,
#   so that models are loaded only once per process.
# These caches are initialized at import time and are updated by helper functions
# in this module; they are internal implementation details and should not be
# modified directly by callers.
_cached_matched_voice = None
_cached_output_lang = None
_piper_voice_cache = {}
_cosyvoice_model_cache = {}
_custom_model_cache = {}


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


def _ensure_cosyvoice_available(verbose=True):
    """Check if CosyVoice is available."""
    if not COSYVOICE_AVAILABLE:
        if verbose:
            print("[CosyVoice][ERROR] CosyVoice is not installed. Install with: pip install -e .[cosyvoice]")
        return False
    return True


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


def cosyvoice_tts(text, model_name, output_wav, reference_audio_path=None, verbose=False):
    """
    Generate speech using CosyVoice TTS.
    
    Parameters
    ----------
    text : str
        Text to synthesize
    model_name : str
        Model name or path (e.g., "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    output_wav : str
        Output WAV file path
    reference_audio_path : str or None
        Path to reference audio for voice cloning
    verbose : bool
        Print debug information
        
    Returns
    -------
    bool
        True if synthesis succeeded, False otherwise
    """
    global _cosyvoice_model_cache
    
    if not _ensure_cosyvoice_available(verbose=verbose):
        return False
    
    try:
        # Default model if none specified
        if not model_name or model_name == "en_US-lessac-medium":
            model_name = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
            
        if verbose:
            print(f"[CosyVoice] Using model: {model_name}")
        
        # Check cache for model
        model = _cosyvoice_model_cache.get(model_name)
        if model is None:
            if verbose:
                print(f"[CosyVoice] Loading model: {model_name}")
            
            # Try to load from local path first, then from HuggingFace
            if os.path.isdir(model_name):
                model = CosyVoice(model_dir=model_name)
            else:
                # Load from HuggingFace or ModelScope
                try:
                    from modelscope import snapshot_download
                    local_dir = os.path.join("pretrained_models", model_name.split("/")[-1])
                    if not os.path.isdir(local_dir):
                        if verbose:
                            print(f"[CosyVoice] Downloading model to {local_dir}")
                        snapshot_download(model_name, local_dir=local_dir)
                    model = CosyVoice(model_dir=local_dir)
                except ImportError:
                    # Fall back to loading directly if modelscope not available
                    if verbose:
                        print(f"[CosyVoice] ModelScope not available, trying direct load")
                    model = CosyVoice(model_dir=model_name)
                    
            _cosyvoice_model_cache[model_name] = model
            if verbose:
                print(f"[CosyVoice] Model loaded successfully")
        else:
            if verbose:
                print(f"[CosyVoice] Reusing cached model")
        
        # Synthesize audio
        if verbose:
            print(f"[CosyVoice] Synthesizing text: '{text[:50]}...'")
        
        # If reference audio is provided, use it for voice cloning
        if reference_audio_path and os.path.isfile(reference_audio_path):
            if verbose:
                print(f"[CosyVoice] Using reference audio: {reference_audio_path}")
            # CosyVoice inference with reference audio (zero-shot cloning)
            output = model.inference_zero_shot(text, reference_audio_path)
        else:
            # Standard TTS inference
            if verbose:
                print(f"[CosyVoice] Using standard TTS (no reference audio)")
            output = model.inference_sft(text)
        
        # Save to WAV file
        # CosyVoice output is typically a tensor or numpy array
        if hasattr(output, 'cpu'):
            # Convert torch tensor to numpy
            audio_data = output.cpu().numpy()
        else:
            audio_data = np.array(output)
        
        # Ensure audio is in correct shape (flatten if needed)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # CosyVoice typically outputs at 22050 Hz
        sample_rate = 22050
        
        # Save as WAV
        sf.write(output_wav, audio_data, sample_rate)
        
        if verbose:
            print(f"[CosyVoice] ✓ Synthesis successful, saved to {output_wav}")
        return True
        
    except Exception as exc:
        if verbose:
            print(f"[CosyVoice][ERROR] TTS synthesis failed: {exc}")
            import traceback
            traceback.print_exc()
        return False


def play_output(translated_text, lang="en", play_audio=True, wav_file=None, rate=16000, voice_backend="gtts", voice_model=None):
    # print(f"[TRACE] play_output called with voice_backend={voice_backend}, voice_model={voice_model}, lang={lang}, play_audio={play_audio}, wav_file={wav_file}, rate={rate}")
    if not translated_text:
        return

    try:
        # Try CosyVoice backend
        if voice_backend == "cosyvoice":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name

            try:
                if cosyvoice_tts(translated_text, voice_model, tts_fp_path, verbose=False):
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
                    return  # Success, exit early
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)
            # If we reach here, CosyVoice failed; fall through to gTTS

        # Try custom backend (Qwen3-TTS)
        elif voice_backend == "custom":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name

            try:
                if custom_tts(translated_text, voice_model, lang, tts_fp_path, verbose=False):
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
                    return  # Success, exit early
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)
            # If we reach here, custom TTS failed; fall through to gTTS

        # Try Piper backend
        elif voice_backend == "piper" and voice_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name

            try:
                if piper_tts(translated_text, voice_model, tts_fp_path, verbose=False):
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
                    return  # Success, exit early
            finally:
                import builtins
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)
            # If we reach here, Piper failed; fall through to gTTS

        # Fallback to gTTS (or primary backend if no other backend specified)
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
    use_cosyvoice = voice_backend == "cosyvoice"
    use_custom = voice_backend == "custom"
    piper_voice = voice_model
    cosyvoice_model = voice_model
    custom_model = voice_model
    lang_base = (output_lang or "en").split("-")[0].split("_")[0].lower()
    
    if not translated_text:
        return None

    tts_pcm = None
    try:
        # CosyVoice backend
        if use_cosyvoice:
            if verbose:
                print(f"[TTS] Using CosyVoice backend")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            try:
                cosyvoice_success = cosyvoice_tts(translated_text, cosyvoice_model, tts_fp_path, verbose=verbose)
                if cosyvoice_success:
                    if verbose:
                        print(f"[TTS] CosyVoice result: SUCCESS")
                    audio_data, sr = sf.read(tts_fp_path)
                    if not audio_data.size:
                        if verbose:
                            print(f"[TTS] CosyVoice result: no data")
                        use_cosyvoice = False
                    else:
                        if sr != rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                            if not audio_data.size:
                                if verbose:
                                    print(f"[TTS] librosa resample result: no data")
                                use_cosyvoice = False
                        if use_cosyvoice and audio_data.size:
                            tts_pcm = (audio_data * 32768).astype(np.int16)
                            if verbose:
                                print(f"[TTS] CosyVoice synthesis complete: {len(tts_pcm)} samples")
                            return tts_pcm
                else:
                    if verbose:
                        print(f"[TTS] CosyVoice failed, falling back to gTTS")
                    use_cosyvoice = False
            except Exception as cosyvoice_exc:
                if verbose:
                    print(f"[TTS] CosyVoice synthesis failed: {cosyvoice_exc}")
                use_cosyvoice = False
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
        
        # Piper backend
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

        if not use_piper and not use_cosyvoice and not use_custom:
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
        TTS backend to use: ``"piper"``, ``"cosyvoice"``, ``"custom"``, or ``"gtts"`` (default: ``"gtts"``)
    voice_model : str or None
        Voice model name for TTS (used as the Piper voice when ``voice_backend`` is ``"piper"``,
        the CosyVoice model when ``voice_backend`` is ``"cosyvoice"``,
        or the Qwen3-TTS model when ``voice_backend`` is ``"custom"``)
    voice_match : bool
        Auto-select Piper voice based on input voice features (for piper backend),
        use reference audio for CosyVoice zero-shot cloning (for cosyvoice backend),
        or use voice cloning with reference audio (for custom backend)
    verbose : bool
        Print debug information
    
    Returns
    -------
    np.ndarray or None
        PCM audio data as int16 array
    
    Notes
    -----
    When ``voice_match`` is enabled:
    - For Piper: performs voice matching only once per process lifetime when reference 
      audio is provided. The matched voice is stored in a module-level cache and reused 
      for later calls to avoid repeated analysis.
    - For CosyVoice: uses reference audio directly for zero-shot voice cloning.
    - For custom backend: the Base model is used with reference audio for voice cloning.
    """ 
  
    
    use_piper = voice_backend == "piper"
    use_cosyvoice = voice_backend == "cosyvoice"
    use_custom = voice_backend == "custom"
    piper_voice = voice_model
    cosyvoice_model = voice_model
    custom_model = voice_model

    if not translated_text:
        return None
    
    try:
        # CosyVoice with voice matching (zero-shot cloning)
        if use_cosyvoice and voice_match and reference_audio is not None:
            if verbose:
                print("==========================================")
                print("COSYVOICE ZERO-SHOT CLONING (--voice-match)")
                print("==========================================")
                print(f"Using reference audio for voice cloning...")
            
            # Save reference audio to a temporary file for CosyVoice
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_fp:
                ref_audio_path = ref_fp.name
                sf.write(ref_audio_path, reference_audio, reference_sample_rate)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_fp:
                tts_fp_path = tts_fp.name
            
            try:
                cosyvoice_success = cosyvoice_tts(
                    translated_text, 
                    cosyvoice_model, 
                    tts_fp_path, 
                    reference_audio_path=ref_audio_path,
                    verbose=verbose
                )
                if cosyvoice_success:
                    if verbose:
                        print(f"✓ CosyVoice zero-shot cloning successful")
                        print("==========================================")
                    audio_data, sr = sf.read(tts_fp_path)
                    if sr != rate:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=rate)
                    tts_pcm = (audio_data * 32768).astype(np.int16)
                    return tts_pcm
                else:
                    if verbose:
                        print("✗ CosyVoice cloning failed, falling back to standard synthesis")
                        print("==========================================")
                    use_cosyvoice = False
            finally:
                import builtins
                if os.path.exists(ref_audio_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(ref_audio_path)
                if os.path.exists(tts_fp_path) and not getattr(builtins, "KEEP_TEMP", False):
                    os.remove(tts_fp_path)
        
        # If CosyVoice without voice matching, use standard synthesis
        if use_cosyvoice:
            tts_pcm = synthesize_tts_pcm(
                translated_text, rate, output_lang,
                voice_backend="cosyvoice",
                voice_model=cosyvoice_model,
                verbose=verbose,
            )
            return tts_pcm
        
        global _cached_matched_voice
        global _cached_output_lang
        
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

        # Standard TTS synthesis (Piper, Custom, or gTTS)
        if use_custom:
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
