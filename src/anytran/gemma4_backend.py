"""Gemma4 multimodal backend for audio transcription and combined transcribe+translate."""

import re
import time

import librosa
import numpy as np

from .config import get_gemma4_config
from .timing import add_timing
from .whisper_backend import is_hallucination


# Phrases indicating the model failed to transcribe or produced an apology
# rather than actual content.
_GEMMA4_SKIP_PHRASES = [
    "unable to transcribe",
    "i cannot transcribe",
    "i can't transcribe",
    "cannot be transcribed",
    "can't be transcribed",
    "i'm not able to transcribe",
]

# Pattern matching timestamp artifacts like "[ 0m0s311ms - 0m1s211ms ]"
_TIMESTAMP_RE = re.compile(
    r"\[\s*\d+m\d+s\d+ms\s*-\s*\d+m\d+s\d+ms\s*\]"
)

# Pattern matching music/sound markers like "[Music]", "[music]", "[ 🎵 ]"
_MUSIC_MARKER_RE = re.compile(
    r"^\s*(?:\[\s*(?:music|🎵|♪|♫)\s*\]\s*)+$",
    re.IGNORECASE,
)

# Pattern matching formatting labels the model prepends to translations, e.g.
# "**French Translation:** ...", "French translation: ...", "**French:** ..."
_LABEL_RE = re.compile(
    r"^\*{0,2}\s*(?:French\s+)?[Tt]ranslat(?:ion|e)\s*:?\s*\*{0,2}\s*:?\s*"
)
_LANG_LABEL_RE = re.compile(
    r"^\*{0,2}\s*French\s*:?\s*\*{0,2}\s*:?\s*"
)

# ---------------------------------------------------------------------------
# Translated prompt echo detection
# ---------------------------------------------------------------------------
# The model sometimes echoes the instruction prompt translated into the target
# language.  For example, the English prompt "Listen to this audio and translate
# it to fr" may appear in French as "Écoutez ceci et traduisez-le en français."
#
# This pattern catches lines containing BOTH a listen/hear-type verb AND a
# translate-type verb in the same line (or transcribe/translate + "audio"), in
# common target languages (en, fr, es, de, it, pt).

_PROMPT_ECHO_RE = re.compile(
    r"(?:"
    # [A] listen/hear verb + ... + translate verb (any common language)
    r"(?:listen|[éè]cout\w*|escuch\w*|ascolt\w*|h[öo]r\w*|ou[çc]\w*)"
    r".{1,80}"
    r"(?:translat\w*|tradui\w*|trad[uú]c\w*|traduz\w*|[üu]bersetz\w*)"
    r"|"
    # [B] translate verb + ... + listen/hear verb (reverse order)
    r"(?:translat\w*|tradui\w*|trad[uú]c\w*|traduz\w*|[üu]bersetz\w*)"
    r".{1,80}"
    r"(?:listen|[éè]cout\w*|escuch\w*|ascolt\w*|h[öo]r\w*|ou[çc]\w*)"
    r"|"
    # [C] transcribe + audio
    r"(?:transcri\w*|transcriv\w*|transkri\w*).{0,40}audio"
    r"|"
    # [D] translate + audio
    r"(?:translat\w*|tradui\w*|trad[uú]c\w*|traduz\w*|[üu]bersetz\w*).{0,40}audio"
    r"|"
    # [E] English meta-instruction leak
    r"(?:output|reply\s+with)\s+only\s+the\s+translat"
    r"|"
    r"do\s+not\s+repeat\s+these\s+instructions"
    r")",
    re.IGNORECASE,
)

# Model apology / inability lines in target languages.
# Catches lines where the model says it cannot listen / transcribe / translate
# the audio, using common apology patterns across languages.
_TRANSLATED_APOLOGY_RE = re.compile(
    r"(?:"
    # sorry/désolé/lo siento + audio/transcribe/listen
    r"(?:sorry|d[ée]sol[ée]\w*|lo\s+siento|mi\s+dispiace|tut\s+mir\s+leid|desculp\w*)"
    r".{0,80}"
    r"(?:audio|transcri\w*|[ée]cout\w*|escuch\w*|ascolt\w*|h[öo]r\w*|ou[çc]\w*)"
    r"|"
    # unable/can't/impossible + transcribe/listen/audio (target languages)
    r"(?:pas\s+pu|ne\s+p(?:eux|eut|ouvons)\s+pas|unable|impossible|kann\s+nicht|non\s+(?:posso|riesco))"
    r".{0,60}"
    r"(?:audio|transcri\w*|[ée]cout\w*|escuch\w*|ascolt\w*|h[öo]r\w*|ou[çc]\w*|tradui\w*|traduc\w*)"
    r")",
    re.IGNORECASE,
)


def _clean_gemma4_output(text, prompt_text=None):
    """Post-process raw Gemma4 model output, removing common artifacts.

    Handles prompt echoes, timestamp markers, model apology lines,
    ``[Music]`` markers, and bold formatting labels that the model sometimes
    injects into its response.

    Parameters
    ----------
    text : str or None
        Raw decoded text from the model.
    prompt_text : str or None
        The prompt that was sent to the model, used to detect prompt echoes.

    Returns
    -------
    str or None
        Cleaned text, or ``None`` if nothing useful remains.
    """
    if not text:
        return None

    prompt_stripped = prompt_text.strip() if prompt_text else None

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # --- skip prompt echoes ------------------------------------------------
        if prompt_stripped and stripped == prompt_stripped:
            continue

        # --- skip translated prompt echoes ------------------------------------
        # Catches the instruction echoed in the target language, e.g.
        # "Écoutez ceci et traduisez-le en français."
        if _PROMPT_ECHO_RE.search(stripped):
            continue

        # --- skip timestamp artifacts (possibly followed by prompt echo) -------
        if _TIMESTAMP_RE.search(stripped):
            remainder = _TIMESTAMP_RE.sub("", stripped).strip()
            # If only a timestamp (possibly plus the prompt), skip
            if not remainder or (prompt_stripped and remainder == prompt_stripped):
                continue
            # Otherwise keep the remainder
            stripped = remainder

        # --- skip model apology / "unable to transcribe" lines -----------------
        lower = stripped.lower()
        if any(phrase in lower for phrase in _GEMMA4_SKIP_PHRASES):
            continue

        # --- skip translated apology lines (target language) -------------------
        if _TRANSLATED_APOLOGY_RE.search(stripped):
            continue

        # --- skip pure music / sound markers -----------------------------------
        if _MUSIC_MARKER_RE.match(stripped):
            continue

        # --- strip inline music markers ----------------------------------------
        stripped = re.sub(
            r"\[[\s🎵♪♫]*(?:music|🎵|♪|♫)[\s🎵♪♫]*\]",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).strip()
        if not stripped:
            continue

        # --- strip formatting labels -------------------------------------------
        stripped = _LABEL_RE.sub("", stripped).strip()
        stripped = _LANG_LABEL_RE.sub("", stripped).strip()

        if stripped:
            cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines).strip()
    return result if result else None

_gemma4_model = None
_gemma4_processor = None
_gemma4_model_id = None


def _get_gemma4_model(model_name=None, verbose=False):
    """Load and cache the Gemma4 model and processor for audio transcription."""
    global _gemma4_model, _gemma4_processor, _gemma4_model_id

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as exc:
        raise ImportError(
            "Gemma4 requires transformers and torch. "
            "Install with: pip install transformers torch accelerate"
        ) from exc

    cfg = get_gemma4_config()
    effective_name = model_name or cfg.get("model_name", "google/gemma-4-E4B-it")

    if _gemma4_model is not None and _gemma4_model_id == effective_name:
        return _gemma4_model, _gemma4_processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Gemma4: Loading processor '{effective_name}' on device '{device}'")
    _gemma4_processor = AutoProcessor.from_pretrained(effective_name, trust_remote_code=True)
    if verbose:
        print(f"Gemma4: Loading model '{effective_name}' on device '{device}'")
    _gemma4_model = AutoModelForImageTextToText.from_pretrained(
        effective_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        _gemma4_model = _gemma4_model.to(device)
    _gemma4_model.eval()
    _gemma4_model_id = effective_name
    if verbose:
        print(f"Gemma4: Loaded model '{effective_name}' on device '{_gemma4_model.device}'")
    return _gemma4_model, _gemma4_processor


def translate_audio_gemma4(
    audio_data,
    samplerate=16000,
    input_lang=None,
    output_lang=None,
    model_name=None,
    verbose=False,
    timers=False,
    timing_stats=None,
):
    """Transcribe (and optionally translate) audio using a Gemma4 multimodal model.

    Parameters
    ----------
    audio_data : array-like
        Audio samples as a NumPy array.
    samplerate : int
        Sample rate of the audio data.
    input_lang : str or None
        Source language hint (or ``None`` for auto-detection).
    output_lang : str or None
        Target language for translation.  When ``None`` or ``"en"``, the
        model is asked to transcribe to English.
    model_name : str or None
        HuggingFace model identifier.  Falls back to the configured default.
    verbose : bool
        Print diagnostic messages.
    timers : bool
        Collect timing information.
    timing_stats : object or None
        Timing aggregator.

    Returns
    -------
    tuple
        ``(audio_data, text, detected_lang)``
    """
    import torch

    timings = [] if timers else None
    t0 = time.perf_counter()

    # Resample to 16 kHz mono float32
    if samplerate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
        samplerate = 16000
    add_timing(timings, "resample", t0)

    t0 = time.perf_counter()
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    add_timing(timings, "preprocess", t0)

    t0 = time.perf_counter()
    model, processor = _get_gemma4_model(model_name=model_name, verbose=verbose)
    add_timing(timings, "model_load", t0)

    # Build the prompt
    target = output_lang or "en"
    if target.lower() in ("en", "eng"):
        prompt_text = (
            "Transcribe this audio to English text. "
            "Reply with ONLY the transcription, nothing else."
        )
    else:
        prompt_text = (
            f"Translate the audio to {target}. "
            "Reply with ONLY the translation. "
            "Do not repeat these instructions or add any commentary."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_data, "sample_rate": samplerate},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    add_timing(timings, "tokenize", t0)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generation = output_ids[0][input_len:]
    raw_text = processor.decode(generation, skip_special_tokens=True).strip()
    add_timing(timings, "generate", t0)

    text = _clean_gemma4_output(raw_text, prompt_text=prompt_text)

    detected_lang = input_lang if input_lang and input_lang.lower() != "auto" else None

    if verbose:
        if raw_text != text:
            print(f"Gemma4 raw output: '{raw_text}'")
        print(f"Gemma4 transcription: '{text}'")
        print(f"Gemma4 detected_lang hint: {detected_lang}")

    if not text:
        if verbose:
            print("Gemma4: No speech detected or transcription failed.")
        return None, None, None

    if is_hallucination(text):
        if verbose:
            print(f"Gemma4: Detected hallucination, skipping: {text}")
        return None, None, None

    if timers and timing_stats is not None:
        timing_stats.add(timings, prefix="gemma4")

    return audio_data, text, detected_lang


def translate_audio_gemma4_combined(
    audio_data,
    samplerate=16000,
    input_lang=None,
    output_lang=None,
    model_name=None,
    verbose=False,
    timers=False,
    timing_stats=None,
):
    """Transcribe and translate audio in a single Gemma4 pass.

    This is used when both scribe-backend and slate-backend are ``gemma4`` with
    the same model, so the English pivot can be skipped entirely.

    Returns
    -------
    tuple
        ``(audio_data, translated_text, detected_lang)``
    """
    import torch

    timings = [] if timers else None
    t0 = time.perf_counter()

    if samplerate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
        samplerate = 16000
    add_timing(timings, "resample", t0)

    t0 = time.perf_counter()
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    add_timing(timings, "preprocess", t0)

    t0 = time.perf_counter()
    model, processor = _get_gemma4_model(model_name=model_name, verbose=verbose)
    add_timing(timings, "model_load", t0)

    target = output_lang or "en"
    prompt_text = (
        f"Translate the audio to {target}. "
        "Reply with ONLY the translation. "
        "Do not repeat these instructions or add any commentary."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_data, "sample_rate": samplerate},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    add_timing(timings, "tokenize", t0)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generation = output_ids[0][input_len:]
    raw_text = processor.decode(generation, skip_special_tokens=True).strip()
    add_timing(timings, "generate", t0)

    text = _clean_gemma4_output(raw_text, prompt_text=prompt_text)

    detected_lang = input_lang if input_lang and input_lang.lower() != "auto" else None

    if verbose:
        if raw_text != text:
            print(f"Gemma4 combined raw output ({target}): '{raw_text}'")
        print(f"Gemma4 combined transcription+translation ({target}): '{text}'")

    if not text:
        if verbose:
            print("Gemma4: No speech detected or transcription/translation failed.")
        return None, None, None

    if is_hallucination(text):
        if verbose:
            print(f"Gemma4: Detected hallucination, skipping: {text}")
        return None, None, None

    if timers and timing_stats is not None:
        timing_stats.add(timings, prefix="gemma4_combined")

    return audio_data, text, detected_lang
