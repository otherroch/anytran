import os
import sys
import subprocess
try:
    from piper.voice import PiperVoice
    PIPER_PYTHON_AVAILABLE = True
except ImportError:
    PiperVoice = None
    PIPER_PYTHON_AVAILABLE = False
import tempfile

import librosa
import numpy as np
import soundfile as sf
from gtts import gTTS
from playsound3 import playsound
from pydub import AudioSegment

from .voice_matcher import (
    extract_voice_features,
    select_best_piper_voice,
)

      
_cached_matched_voice = None
_piper_voice = None

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
    
    global _piper_voice
    
    #if verbose:
    #    print(f"[TRACE] piper_tts called with voice_model={voice_model}, output_wav={output_wav}, verbose={verbose}")
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
        if verbose:
            print(f"[Piper] Using model path: {model_path}")
            print(f"[Piper] Model file exists: {os.path.isfile(model_path)}")
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

        # Use Piper Python API
        try:
            voice = None
            if _piper_voice is not None:
                voice = _piper_voice
                if verbose:
                    print(f"[Piper] Reusing cached PiperVoice instance for model: {voice_model}")
            else:
                voice = PiperVoice.load(model_path)
                _piper_voice = voice
                if verbose:
                    print(f"[Piper] Loaded new PiperVoice instance for model: {voice_model}")
            if verbose:
                print(f"[Piper] Synthesizing with PiperVoice.synthesize_wav...")
            import wave
            with wave.open(output_wav, "wb") as wav_file:
                # PiperVoice.synthesize_wav sets the format automatically
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
            tts_lang = lang.split("-")[0]
            tts = gTTS(text=translated_text, lang=tts_lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_fp:
                tts.save(tts_fp.name)
                tts_fp_path = tts_fp.name
            try:
                if play_audio:
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
    use_piper = voice_backend == "piper"
    piper_voice = voice_model
    
    if not translated_text:
        return None

    tts_pcm = None
    try:
        # For non-English output, prefer gTTS over Piper if no explicit voice given
        # gTTS has better language support and voice diversity
        lang_base = (output_lang or "en").split("-")[0]
        piper_has_limited_voices = lang_base.lower() in ["it", "pt"]  # Languages with very limited Piper voices
        
        # If Piper is requested but language has limited voices, prefer gTTS instead
        if use_piper and piper_voice and piper_has_limited_voices:
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

        if not use_piper:
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
        voice_backend="gtts",
        voice_model=None,
        voice_match=False,
        verbose=False
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
    voice_backend : str
        TTS backend to use, either ``"piper"`` or ``"gtts"`` (default: ``"gtts"``)
    voice_model : str or None
        Voice model name for TTS (used as the Piper voice when ``voice_backend`` is ``"piper"``)
    voice_match : bool
        Auto-select Piper voice based on input voice features
    verbose : bool
        Print debug information
    
    Returns
    -------
    np.ndarray or None
        PCM audio data as int16 array
    
    Notes
    -----
    When ``voice_match`` is enabled, this function performs voice matching
    only once per process lifetime. The first call that requires automatic
    voice matching selects the best Piper voice based on the input voice
    features and caches that selection in the module-level
    ``_cached_matched_voice`` variable. Subsequent calls in the same process
    reuse the cached match instead of re-running voice matching, unless the
    process is restarted.
    """ 
    global _cached_matched_voice
    use_piper = voice_backend == "piper"
    piper_voice = voice_model

    if not translated_text:
        return None
    
    try:
        # Auto voice matching with Piper
        # Only apply if user didn't explicitly specify a non-default voice
        # Default voice is "en_US-lessac-medium"
        explicit_voice_provided = piper_voice and piper_voice != "en_US-lessac-medium"
  
        cached_auto_match_applicable = voice_match and reference_audio is not None and not explicit_voice_provided
        if cached_auto_match_applicable and _cached_matched_voice is None:
            if verbose:
                print("==========================================")
                print("AUTO-VOICE MATCHING (--voice-match)")
                print("==========================================")
                print(f"Analyzing input voice characteristics...")
            
            # Extract voice features
            features = extract_voice_features(reference_audio, reference_sample_rate, verbose=True if verbose else False)
            
            # Select best matching Piper voice
            if verbose:
                print(f"Selecting best {output_lang} voice for detected characteristics...")
            matched_voice = select_best_piper_voice(features, output_lang, verbose=True if verbose else False)
            
            if matched_voice:
                if verbose:
                    print(f"✓ Voice matching successful: {matched_voice}")
                    print("==========================================")
                piper_voice = matched_voice
                _cached_matched_voice = matched_voice
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
                print(f"Using previously matched voice: {_cached_matched_voice}")
            piper_voice = _cached_matched_voice
            use_piper = True    
        
        
        # Standard Piper TTS or gTTS
        return synthesize_tts_pcm(
            translated_text, rate, output_lang,
            voice_backend="piper" if use_piper else "gtts",
            voice_model=piper_voice,
            verbose=verbose,
        )
        
    except Exception as exc:
        if verbose:
            print(f"TTS synthesis with voice features failed: {exc}")
        return None
