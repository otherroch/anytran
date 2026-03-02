import numpy as np
import time

from .mqtt_client import send_mqtt_text
from .text_translator import translate_text
from .timing import add_timing, format_timing
from .tts import play_output, synthesize_tts_pcm, synthesize_tts_pcm_with_cloning
from .utils import normalize_lang_code
from .vad import SILERO_AVAILABLE, has_speech_silero
from .whisper_backend import translate_audio


def build_output_prefix(stream_id=None, detected_lang=None):
    lang_map = {
        "en": "English",
        "fr": "French",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
        "vi": "Vietnamese",
        "th": "Thai",
        "tr": "Turkish",
        "nl": "Dutch",
        "sv": "Swedish",
        "no": "Norwegian",
        "da": "Danish",
        "fi": "Finnish",
        "pl": "Polish",
        "cs": "Czech",
        "el": "Greek",
        "he": "Hebrew",
        "id": "Indonesian",
        "ms": "Malay",
        "uk": "Ukrainian",
    }
    if not detected_lang:
        return "Unknown: "
    normalized = normalize_lang_code(detected_lang)
    if not normalized:
        return "Unknown: "
    name = lang_map.get(normalized)
    if not name:
        base = normalized.split("-")[0]
        name = lang_map.get(base)
    if not name:
        return "Unknown: "
    return f"{name}: "


def process_audio_chunk(
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
    mqtt_topic,
    stream_id=None,
    scribe_vad=True,
    voice_backend="gtts",
    voice_model=None,
    chat_logger=None,
    rtsp_ip=None,
    timers=False,
    timing_stats=None,
    scribe_backend="auto",
    text_translation_target=None,
    slate_backend="googletrans",
    voice_lang=None,
    scribe_text_file=None,
    slate_text_file=None,
    scribe_tts_segments=None,
    slate_tts_segments=None,
    langswap_enabled=False,
    langswap_input_lang=None,
    langswap_output_lang=None,
    voice_match=False,
    lang_prefix=False,
):
    """
    Process an audio chunk through a 3-stage pipeline:
    
    **Stage 1 (Transcription):** Voice audio → English text
        - Uses Whisper backend to transcribe/translate audio to English
        - Output: English text (transcription)
        
    **Stage 2 (Translation):** English text → target language text
        - Only if text_translation_target is specified and != "en"
        - Uses Google Translate, LibreTranslate, or local AI model
        - Output: Translated text in target language
        
    **Stage 3 (TTS):** Text → Voice audio
        - Only if TTS output is requested (scribe_tts_segments or slate_tts_segments)
        - Uses gTTS or Piper to synthesize speech
        - Output: Audio in appropriate language

    This function is safe to call from both streaming/web contexts and non-web contexts.
    All side-effecting outputs (TTS, MQTT, logging, etc.) are controlled via arguments.

    Parameters
    ----------
    audio_segment : np.ndarray
        1‑D array of raw PCM audio samples (typically int16 or float) for the chunk
        to process.
    rate : int
        Sample rate in Hz for ``audio_segment``.
    input_lang : str or None
        Optional input language code (e.g. ``"en"``, ``"de"``, ``"auto"``). If
        ``None`` or ``"auto"``, the backend may perform language detection.
    output_lang : str or None
        Target language code for translation or transcription. If ``None``, the
        backend may choose a sensible default (often the detected input language).
    magnitude_threshold : float
        Minimum mean absolute magnitude of the audio chunk required to consider it
        as non‑silence before applying optional VAD.
    model : str or None
        Name of the speech/translation model to use. If ``None``, a default
        (currently ``"medium"``) is used.
    verbose : bool
        If ``True``, print diagnostic information about magnitude, VAD, and
        translation progress.
    mqtt_broker : str or None
        Hostname or IP address of the MQTT broker. If ``None``, no MQTT messages are
        published.
    mqtt_port : int or None
        Port number for the MQTT broker. Ignored if ``mqtt_broker`` is ``None``.
    mqtt_username : str or None
        Optional username for authenticating with the MQTT broker.
    mqtt_password : str or None
        Optional password for authenticating with the MQTT broker.
    mqtt_topic : str or None
        Topic to which translated text should be published. If ``None`` or empty,
        MQTT publishing is skipped even if a broker is configured.
    stream_id : str or int or None, optional
        Identifier for the current audio stream or session. Used only for logging
        and debugging output; may be ``None`` if not needed.
    scribe_vad : bool, optional
        If ``True`` and Silero VAD is available, run voice activity detection to
        further filter out non‑speech segments after the magnitude check.
    voice_backend : str, optional
        TTS backend to use, either ``"piper"`` or ``"gtts"`` (default: ``"gtts"``).
    voice_model : str or None, optional
        Voice model name for TTS. Used as the Piper voice when ``voice_backend`` is
        ``"piper"``.
    chat_logger : callable or object or None, optional
        Optional logger used to record recognized/translated text for chat or UI
        purposes. This is typically a callable taking the formatted text (and
        possibly additional context), or an object exposing an equivalent logging
        method. If ``None``, no chat logging is performed.
    rtsp_ip : str or None, optional
        Optional IP or URL associated with an RTSP source. Used only for contextual
        labeling or logging; does not affect translation logic.
    timers : bool, optional
        If ``True``, collect timing information for different processing stages
        (magnitude, VAD, translation, TTS, etc.) and store it in ``timing_stats``.
    timing_stats : dict or None, optional
        Optional dictionary used to accumulate timing statistics across multiple
        calls. When provided and ``timers`` is ``True``, per‑stage timings are added
        via :func:`add_timing`.
    scribe_backend : str, optional
        Preference for the whisper/transcription backend, e.g. ``"auto"`` to let the
        system choose, or a specific backend name if multiple implementations are
        available.
    text_translation_target : str or None, optional
        Target language code for Stage 2 text-to-text translation. If ``None`` or
        ``"en"``, Stage 2 is skipped and English transcription is used directly.
    slate_backend : str, optional
        Backend for Stage 2 translation: ``"googletrans"``, ``"libretranslate"``,
        or ``"none"``. Default is ``"googletrans"``.
    voice_lang : str or None, optional
        Override for Stage 3 TTS language. If ``None``, uses ``text_translation_target``
        or falls back to ``output_lang``.

    Returns
    -------
    str or None
        The formatted output text (including a language prefix as produced by
        :func:`build_output_prefix`) from the final stage of the pipeline.
        
        - If Stage 2 ran: Returns translated text in target language
        - If Stage 2 skipped: Returns English transcription from Stage 1
        - Returns ``None`` if the chunk is silence or translation fails

    Side Effects
    ------------
    - May perform VAD to skip non-speech audio
    - May synthesize TTS audio and append to ``scribe_tts_segments`` or ``slate_tts_segments``
    - May publish translated text over MQTT (if ``mqtt_broker`` configured)
    - May log text via ``chat_logger`` (if provided)
    - May update ``timing_stats`` with per-stage timing information (if ``timers=True``)
    
    Notes
    -----
    Timing keys when ``timers=True``:
    - ``magnitude``: Magnitude calculation
    - ``vad``: Voice activity detection (if enabled)
    - ``stage1_transcription``: Whisper audio transcription to English
    - ``stage2_translation``: Text-to-text translation (if target != en)
    - ``stage3_tts_synthesis``: TTS audio synthesis (if requested)
    - ``stage3_tts_playback``: TTS audio playback (if requested)
    - ``text_out``, ``mqtt``, ``chat_log``, ``tts_append``: Output operations
    """
    timings = [] if timers else None
    t0 = time.perf_counter()
    magnitude = np.abs(audio_segment).mean()
    if verbose:
        prefix = f"[Stream {stream_id}] " if stream_id else ""
        print(f"{prefix}Input audio magnitude: {magnitude}")
    has_speech = magnitude >= magnitude_threshold
    add_timing(timings, "magnitude", t0)

    if has_speech and scribe_vad and SILERO_AVAILABLE:
        t0 = time.perf_counter()
        has_speech = has_speech_silero(audio_segment, rate)
        if verbose:
            prefix = f"[Stream {stream_id}] " if stream_id else ""
            print(f"{prefix}Using Silero VAD for speech detection.")
        add_timing(timings, "vad", t0)

    if not has_speech:
        if verbose:
            prefix = f"[Stream {stream_id}] " if stream_id else ""
            print(f"{prefix}Silence detected, skipping...")
        return None

    # ============================================================================
    # STAGE 1: VOICE TRANSCRIPTION TO ENGLISH TEXT
    # ============================================================================
    # Uses Whisper backend (whispercpp, faster-whisper, or whisper-ctranslate2)
    # to transcribe/translate audio to English text.
    # Output: english_text (Stage 1 result)
    
    prefix = f"[Stream {stream_id}] " if stream_id else ""
    
    # IMPORTANT: When LangSwap is enabled, force auto-detection
    # We need Whisper to naturally detect the language without bias
    # so LangSwap can determine the correct translation direction
    stage1_input_lang = input_lang
    if langswap_enabled and langswap_input_lang and langswap_output_lang:
        stage1_input_lang = None  # Force auto-detection
        if verbose:
            print(f"{prefix}LangSwap Pre-Stage1: Forcing auto-detection (ignoring input_lang hint)")
            print(f"{prefix}  - Original input_lang: {input_lang}")
            print(f"{prefix}  - Using for Whisper: None (auto-detect)")
    
    if verbose:
        print(f"{prefix}Stage 1 (Whisper Transcription): Starting")
        print(f"{prefix}  - Input language hint: {stage1_input_lang or 'auto-detect'}")
        print(f"{prefix}  - Output language: {output_lang or 'en'}")
        print(f"{prefix}  - Backend: {scribe_backend}")
    
    model_name = model if model else "medium"
    t0 = time.perf_counter()
    output_audio_data, english_text, detected_lang = translate_audio(
        audio_segment,
        rate,
        stage1_input_lang,  # Use potentially modified input_lang
        output_lang,
        model=model_name,
        backend_preference=scribe_backend,
        verbose=verbose,
        timers=timers,
        timing_stats=timing_stats,
    )
    add_timing(timings, "stage1_transcription", t0)
    
    if not english_text:
        return None
    
    if verbose:
        print(f"{prefix}Stage 1 (Transcription): '{english_text}'")
        print(f"{prefix}  - Detected language: {detected_lang}")
        print(f"{prefix}  - Transcription complete")

    # ============================================================================
    # LANGSWAP: AUTOMATIC LANGUAGE DETECTION AND TARGET SWITCHING
    # ============================================================================
    # If LangSwap is enabled, automatically determine the translation target
    # based on the detected input language. This enables bidirectional translation
    # where both input and output languages can be spoken, and the system
    # automatically translates to the opposite language.
    
    prefix = f"[Stream {stream_id}] " if stream_id else ""
    langswap_changed_target = False  # Track if langswap changed the translation target
    
    if langswap_enabled and langswap_input_lang and langswap_output_lang:
        # Normalize detected language for comparison
        detected_base = normalize_lang_code(detected_lang)
        input_base = normalize_lang_code(langswap_input_lang)
        output_base = normalize_lang_code(langswap_output_lang)
        
        print(f"{prefix}========================================================")
        print(f"{prefix}LANGSWAP ENABLED - Bidirectional Translation Active")
        print(f"{prefix}========================================================")
        print(f"{prefix}Detected Language:    {detected_lang} -> normalized: {detected_base}")
        print(f"{prefix}Input Language:       {langswap_input_lang} -> normalized: {input_base}")
        print(f"{prefix}Output Language:      {langswap_output_lang} -> normalized: {output_base}")
        print(f"{prefix}Original Target:      {text_translation_target}")
        print(f"{prefix}========================================================")
        
        # Store original target for comparison
        original_target = text_translation_target
        
        # Determine translation target based on detected language
        if detected_base and input_base and detected_base == input_base:
            # Detected input language, translate to output language
            text_translation_target = langswap_output_lang
            langswap_changed_target = True
            print(f"{prefix}DECISION: Detected language '{detected_base}' matches INPUT language '{input_base}'")
            print(f"{prefix}ACTION:   Translating FROM {input_base} TO {langswap_output_lang}")
            print(f"{prefix}METHOD:   Stage 1 (Whisper) -> Stage 2 (Text Translation)")
        elif detected_base and output_base and detected_base == output_base:
            # Detected output language, translate to input language
            text_translation_target = langswap_input_lang
            langswap_changed_target = True
            print(f"{prefix}DECISION: Detected language '{detected_base}' matches OUTPUT language '{output_base}'")
            print(f"{prefix}ACTION:   Translating FROM {output_base} TO {langswap_input_lang}")
            print(f"{prefix}METHOD:   Stage 1 (Whisper already translated to EN) -> Stage 2 (if needed)")
        else:
            # Could not determine which language, keep original target
            print(f"{prefix}DECISION: Detected language '{detected_base}' does NOT match either:")
            print(f"{prefix}           - Input: '{input_base}'")
            print(f"{prefix}           - Output: '{output_base}'")
            print(f"{prefix}ACTION:   Using original translation target: {text_translation_target}")
            print(f"{prefix}NOTE:     This may indicate unexpected language detection")
        
        print(f"{prefix}New Translation Target: {text_translation_target}")
        print(f"{prefix}========================================================")
    elif langswap_enabled:
        print(f"{prefix}WARNING: LANGSWAP ENABLED but missing configuration:")
        print(f"{prefix}  - Input Lang: {langswap_input_lang}")
        print(f"{prefix}  - Output Lang: {langswap_output_lang}")
        print(f"{prefix}  LangSwap requires both languages to be explicitly set (not 'auto')")

    # ============================================================================
    # STAGE 2: TEXT-TO-TEXT TRANSLATION (English → Target Language)
    # ============================================================================
    # Only runs if text_translation_target is specified and is not English.
    # Uses googletrans, LibreTranslate, or DeepL for translation.
    # Output: translated_text (Stage 2 result), or english_text if Stage 2 skipped
    
    translated_text = english_text  # Default: no translation
    stage2_ran = False
    
    prefix = f"[Stream {stream_id}] " if stream_id else ""
    
    if text_translation_target and text_translation_target.lower().split("-")[0] != "en":
        if verbose:
            print(f"{prefix}Stage 2 (Text Translation): Preparing to translate")
            print(f"{prefix}  - Source text: '{english_text}'")
            print(f"{prefix}  - Source language: en (from Whisper)")
            print(f"{prefix}  - Target language: {text_translation_target}")
            print(f"{prefix}  - Translation backend: {slate_backend}")
        
        t0 = time.perf_counter()
        text_translated = translate_text(
            english_text,
            source_lang="en",
            target_lang=text_translation_target,
            backend=slate_backend,
            verbose=verbose
        )
        if text_translated:
            translated_text = text_translated
            stage2_ran = True
            if verbose:
                print(f"{prefix}Stage 2 (Translation en→{text_translation_target}): {translated_text}")
        else:
            if verbose:
                print(f"{prefix}Stage 2 (Translation): FAILED - using English text")
        add_timing(timings, "stage2_translation", t0)
    else:
        if verbose:
            if text_translation_target:
                print(f"{prefix}Stage 2 (Translation): SKIPPED - target is English ({text_translation_target})")
                print(f"{prefix}  - Output will be: '{english_text}'")
            else:
                print(f"{prefix}Stage 2 (Translation): SKIPPED - no translation target set")

    # Determine final text and language for downstream processing
    final_text = translated_text
    
    # Determine the final output language:
    # 1. If Stage 2 ran, use the translation target
    # 2. If a translation target was set (including English), prefer it
    # 3. Otherwise, use the detected language
    if stage2_ran:
        # Stage 2 translation happened, output is in text_translation_target
        final_text_lang = text_translation_target
    elif text_translation_target:
        # Translation skipped (e.g., target English) but target still defines output language
        final_text_lang = text_translation_target
    else:
        # No translation occurred, use detected language from Stage 1
        final_text_lang = detected_lang
    
    prefix = f"[Stream {stream_id}] " if stream_id else ""
    
    if verbose:  # Only print when verbose is True
        print(f"{prefix}========================================================")
        print(f"{prefix}LANGUAGE DECISION:")
        print(f"{prefix}  stage2_ran={stage2_ran}, langswap_changed_target={langswap_changed_target}")
        print(f"{prefix}  text_translation_target={text_translation_target}")
        print(f"{prefix}  detected_lang={detected_lang}")
        print(f"{prefix}  -> final_text_lang={final_text_lang}")
        print(f"{prefix}========================================================")
        print(f"{prefix}PIPELINE SUMMARY:")
        print(f"{prefix}  Stage 1 Output (Whisper): '{english_text}' [lang: {detected_lang}]")
        if stage2_ran:
            print(f"{prefix}  Stage 2 Output (Translation): '{translated_text}' [lang: {text_translation_target}]")
        else:
            print(f"{prefix}  Stage 2: SKIPPED")
        print(f"{prefix}  Final Output: '{final_text}' [lang: {final_text_lang}]")
        print(f"{prefix}========================================================")

    # ============================================================================
    # STAGE 3: TEXT-TO-SPEECH (TTS)
    # ============================================================================
    # Synthesizes voice audio from final_text if TTS output is requested.
    # Uses gTTS or Piper. Language determined by voice_lang parameter or final_text_lang.
    
    # Determine TTS language:
    # 1. Priority: voice_lang parameter (explicit override)
    # 2. Final text language (from Stage 2 translation target or detected language)
    # 3. Fallback: output_lang
    # NOTE: The final_text_lang is critical for LangSwap to work correctly,
    # as it captures the actual language of the translated output.
    tts_lang = voice_lang or final_text_lang or output_lang
    
    if verbose:
        prefix = f"[Stream {stream_id}] " if stream_id else ""
        print(f"{prefix}Stage 3 (TTS): Preparing voice synthesis")
        print(f"{prefix}  - TTS language: {tts_lang}")
        print(f"{prefix}    (from: voice_lang={voice_lang}, final_text_lang={final_text_lang}, output_lang={output_lang})")

    # TTS synthesis for both scribe (English) and slate (translated) audio
    scribe_tts_pcm = None
    slate_tts_pcm = None
    
    # Synthesize scribe audio (English)
    if english_text and scribe_tts_segments is not None:
        t0 = time.perf_counter()
        scribe_tts_pcm = synthesize_tts_pcm_with_cloning(
            english_text,
            rate,
            "en",
            reference_audio=audio_segment if voice_match else None,
            reference_sample_rate=rate,
            voice_backend=voice_backend,
            voice_model=voice_model,
            voice_match=voice_match,
            verbose=verbose,
        )
        add_timing(timings, "stage3_tts_synthesis", t0)
        if verbose:
            prefix = f"[Stream {stream_id}] " if stream_id else ""
            print(f"{prefix}Stage 3 (TTS - Scribe/English): Generated voice audio")
    
    # Synthesize slate audio (final output)
    if final_text and slate_tts_segments is not None:
        t0 = time.perf_counter()
        slate_tts_pcm = synthesize_tts_pcm_with_cloning(
            final_text,
            rate,
            tts_lang,
            reference_audio=audio_segment if voice_match else None,
            reference_sample_rate=rate,
            voice_backend=voice_backend,
            voice_model=voice_model,
            voice_match=voice_match,
            verbose=verbose,
        )
        add_timing(timings, "stage3_tts_synthesis", t0)
        if verbose:
            prefix = f"[Stream {stream_id}] " if stream_id else ""
            print(f"{prefix}Stage 3 (TTS - Slate/{tts_lang}): Generated voice audio")
    
    # Legacy: tts_segments removed

    # Play audio output (if requested)
    # play_audio removed

    # ============================================================================
    # OUTPUT FORMATTING AND DISTRIBUTION
    # ============================================================================
    # Format final text with language prefix and distribute to all requested outputs:
    # - Text file
    # - MQTT broker
    # - Chat logger
    # - TTS segments buffer
    
    # Build output text, optionally with language prefix
    output_text = None
    if final_text:
        if lang_prefix:
            # Use the language of the final text (translated if Stage 2 ran, otherwise detected language)
            prefix = build_output_prefix(stream_id=stream_id, detected_lang=final_text_lang)
            output_text = f"{prefix}{final_text.strip()}"
        else:
            output_text = final_text.strip()

    # text_file removed

    # Send to MQTT broker (if configured)
    if mqtt_broker and output_text:
        t0 = time.perf_counter()
        send_mqtt_text(output_text, mqtt_topic, mqtt_broker, mqtt_port, mqtt_username, mqtt_password)
        add_timing(timings, "mqtt", t0)

    # Log to chat logger (if configured)
    if chat_logger and output_text and rtsp_ip:
        t0 = time.perf_counter()
        chat_logger.log(rtsp_ip, output_text)
        print(f"Logged to chat log:  {rtsp_ip} =>-{output_text}")
        add_timing(timings, "chat_log", t0)

    # Append TTS audio segments (if requested)
    # New: Separate scribe and slate audio
    if scribe_tts_segments is not None and scribe_tts_pcm is not None:
        t0 = time.perf_counter()
        scribe_tts_segments.append(scribe_tts_pcm)
        add_timing(timings, "tts_append", t0)
    
    if slate_tts_segments is not None and slate_tts_pcm is not None:
        t0 = time.perf_counter()
        slate_tts_segments.append(slate_tts_pcm)
        add_timing(timings, "tts_append", t0)
    
    # Legacy: tts_segments removed

    # Print timing information (if requested)
    if timers:
        # prefix = f"[Stream {stream_id}] " if stream_id else ""
        # print(f"{prefix}Timing chunk: {format_timing(timings)}")
        if timing_stats is not None:
            timing_stats.add(timings, prefix="chunk")

    # Return stage outputs for runner deduplication
    scribe_output = None
    slate_output = None
    if english_text:
        if lang_prefix:
            scribe_output = f"{build_output_prefix(stream_id=stream_id, detected_lang='en')}{english_text.strip()}"
        else:
            scribe_output = english_text.strip()
    if stage2_ran and translated_text:
        if lang_prefix:
            slate_output = f"{build_output_prefix(stream_id=stream_id, detected_lang=final_text_lang)}{translated_text.strip()}"
        else:
            slate_output = translated_text.strip()
    
    return {
        'output': output_text,
        'scribe': scribe_output,
        'slate': slate_output,
        'final_lang': final_text_lang,  # NEW: Return actual final language for TTS purposes
    }
