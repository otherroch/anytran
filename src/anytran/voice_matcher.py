"""
Voice matching utilities for matching output voice to input voice characteristics.

This module provides voice feature analysis and matching to select nearest Piper voice.
"""

import numpy as np
import librosa
import tempfile
import os
import json
from pathlib import Path
from functools import lru_cache


VOICE_TABLE_JSON_PATH = Path(__file__).resolve().parent / "voice_table.json"


def extract_voice_features(audio_data, sample_rate=16000, verbose=False):
    """
    Extract voice characteristics from audio for matching.
    
    Parameters
    ----------
    audio_data : np.ndarray
        Audio samples as float32 array
    sample_rate : int
        Sample rate in Hz
    verbose : bool
        Print debug information
    
    Returns
    -------
    dict
        Dictionary containing:
        - mean_pitch: Average fundamental frequency in Hz
        - pitch_std: Standard deviation of pitch
        - zcr: Zero crossing rate (proxy for speaking rate)
        - brightness: Spectral centroid (voice brightness)
        - gender: Estimated gender ("male" or "female")
        - voice_type: Categorized voice type
    """
    if verbose:
        print(f"Extracting voice features from audio ({len(audio_data)} samples)...")
    
    # Ensure float32 and normalize
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / 32768.0
    
    # Extract fundamental frequency (pitch) using YIN algorithm
    try:
        f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=sample_rate)
        # Filter out unvoiced frames (NaN or very low values)
        f0_valid = f0[~np.isnan(f0)]
        f0_valid = f0_valid[f0_valid > 50]
        
        if len(f0_valid) > 0:
            mean_pitch = np.mean(f0_valid)
            pitch_std = np.std(f0_valid)
        else:
            mean_pitch = 150.0  # Default neutral pitch
            pitch_std = 0.0
    except Exception as e:
        if verbose:
            print(f"Pitch extraction failed: {e}, using default")
        mean_pitch = 150.0
        pitch_std = 0.0
    
    # Zero crossing rate (speaking rate proxy)
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    mean_zcr = np.mean(zcr)
    
    # Spectral centroid (brightness/timbre)
    try:
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        )[0]
        mean_brightness = np.mean(spectral_centroid)
    except Exception:
        mean_brightness = 2000.0
    
    # Gender estimation based on pitch
    # Typical ranges: Male 85-200 Hz, Female 165-255 Hz
    # Uses brightness as tiebreaker for ambiguous range (150-190 Hz)
    if mean_pitch < 150:
        gender = "male"
        if mean_pitch < 100:
            voice_type = "male_deep"
        else:
            voice_type = "male_mid"
    elif mean_pitch < 190:
        # Ambiguous range (150-190 Hz), use brightness as secondary indicator
        # High brightness = more likely female, low = more likely male
        if mean_brightness > 2500:
            gender = "female"
            voice_type = "female_low"
        else:
            gender = "male"
            voice_type = "male_mid"
    else:
        gender = "female"
        if mean_pitch > 220:
            voice_type = "female_high"
        else:
            voice_type = "female_mid"
    
    features = {
        "mean_pitch": float(mean_pitch),
        "pitch_std": float(pitch_std),
        "zcr": float(mean_zcr),
        "brightness": float(mean_brightness),
        "gender": gender,
        "voice_type": voice_type
    }
    
    if verbose:
        print(f"Voice features: pitch={mean_pitch:.1f}Hz (±{pitch_std:.1f}), brightness={mean_brightness:.0f}Hz, gender={gender}, type={voice_type}")
    
    return features


def select_best_piper_voice(features, language="en", verbose=False):
    """
    Select the best matching Piper voice based on extracted features.
    
    Parameters
    ----------
    features : dict
        Voice features from extract_voice_features()
    language : str
        Target language code (e.g., "en", "fr", "es")
    verbose : bool
        Print selection reasoning
    
    Returns
    -------
    str or None
        Piper voice model name (e.g., "en_US-ryan-high") or None if no match
    """
    piper_voices = _load_piper_voices()
    
    # Normalize language code (handle en-US -> en, etc.)
    lang_base = language.split("-")[0].split("_")[0]
    lang_voices = piper_voices.get(lang_base) or piper_voices.get("en", {})
    if not lang_voices:
        return None
    
    # Find closest match by voice attributes and gender
    best_voice = None
    gender_match_min_score = float('inf')
    fallback_voice = None
    fallback_min_score = float('inf')
    
    if verbose:
        print(f"Searching for {language} voices matching: gender={features['gender']}, pitch={features['mean_pitch']:.1f}Hz")
    
    # First pass: Find best match with same gender
    for voice_name, voice_data in lang_voices.items():
        match_score = _voice_match_score(features, voice_data)
        pitch_diff = abs(features["mean_pitch"] - voice_data["pitch"]) if verbose else None
        # Track fallback option (closest score regardless of gender)
        if match_score < fallback_min_score:
            fallback_min_score = match_score
            fallback_voice = voice_name
        
        # Prioritize gender match
        if features["gender"] == voice_data["gender"]:
            if match_score < gender_match_min_score:
                gender_match_min_score = match_score
                best_voice = voice_name
                if verbose:
                    print(
                        f"  - {voice_name}: gender={voice_data['gender']}, pitch={voice_data['pitch']}Hz "
                        f"(diff={pitch_diff:.1f}Hz, score={match_score:.2f}) ✓ MATCH"
                    )
            elif verbose:
                print(
                    f"  - {voice_name}: gender={voice_data['gender']}, pitch={voice_data['pitch']}Hz "
                    f"(diff={pitch_diff:.1f}Hz, score={match_score:.2f})"
                )
        elif verbose:
            print(
                f"  - {voice_name}: gender={voice_data['gender']}, pitch={voice_data['pitch']}Hz "
                f"(diff={pitch_diff:.1f}Hz, score={match_score:.2f}) [no gender match]"
            )
    
    # Use fallback if no gender match found
    if best_voice is None and fallback_voice is not None:
        best_voice = fallback_voice
        fallback_data = lang_voices.get(fallback_voice, {})
        if verbose:
            print(f"\n⚠ No {features['gender']} voice available for {language}")
            print(f"Using fallback: {fallback_voice} ({fallback_data.get('gender')} voice, score: {fallback_min_score:.2f})")
            if len(lang_voices) == 1:
                print(f"Note: {language} has only one available voice in Piper")
    
    if verbose and best_voice:
        selected_data = lang_voices.get(best_voice, {})
        print(f"\n✓ Selected Piper voice: {best_voice} (gender={selected_data.get('gender')}, pitch={selected_data.get('pitch')}Hz)")
    
    return best_voice


def _voice_match_score(features, voice_data):
    score = abs(features["mean_pitch"] - voice_data["pitch"]) / 30.0

    if "pitch_std" in features and voice_data.get("pitch_std") is not None:
        score += abs(features["pitch_std"] - voice_data["pitch_std"]) / 20.0
    if "zcr" in features and voice_data.get("zcr") is not None:
        score += abs(features["zcr"] - voice_data["zcr"]) / 0.05
    if "brightness" in features and voice_data.get("brightness") is not None:
        score += abs(features["brightness"] - voice_data["brightness"]) / 500.0

    return score


@lru_cache(maxsize=1)
def _load_piper_voices():
    try:
        with open(VOICE_TABLE_JSON_PATH, "r", encoding="utf-8") as f:
            table = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}

    piper_voices = {}
    for entry in table if isinstance(table, list) else []:
        onnx_file = entry.get("onnx_file")
        pitch = entry.get("pitch")
        gender = entry.get("gender")
        if not onnx_file or pitch is None or not gender:
            continue
        voice_name = os.path.splitext(os.path.basename(onnx_file))[0]
        lang_base = voice_name.split("-")[0].split("_")[0].lower()
        piper_voices.setdefault(lang_base, {})[voice_name] = {
            "pitch": pitch,
            "gender": gender,
            "pitch_std": entry.get("pitch_std"),
            "zcr": entry.get("zcr"),
            "brightness": entry.get("brightness"),
        }
    return piper_voices
