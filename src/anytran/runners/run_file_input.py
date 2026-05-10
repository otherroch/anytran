"""File-input runner.

Reads a local audio or text file, runs the translation pipeline,
and writes output audio/text to disk.
"""

import os
import time

import numpy as np

from ..pipeline_config import RunnerConfig, StreamContext
from ..stream_output import compute_window_params, normalize_text, output_audio


def run_file_input(
    input_path: str,
    cfg: RunnerConfig,
):
    """Run the pipeline on a local audio or text file.

    Parameters
    ----------
    input_path : str
        Path to the input file (audio ``.wav``, ``.mp3``, etc. or text ``.txt``).
    cfg : RunnerConfig
        Combined runner configuration replacing ~30 individual keyword
        parameters.
    """
    pipeline = cfg.pipeline
    output = cfg.output
    mqtt = cfg.mqtt

    # -- output paths -------------------------
    scribe_text_file = output.scribe_text_file
    slate_text_file = output.slate_text_file
    output_audio_path = output.output_audio_path
    slate_audio_path = output.slate_audio_path
    capture_voice_path = output.capture_voice_path

    # -- pipeline settings --------------------
    verbose = pipeline.verbose
    timers_all = pipeline.timers_all
    keep_temp = pipeline.keep_temp
    input_lang = pipeline.input_lang
    text_translation_target = pipeline.text_translation_target
    slate_backend = pipeline.slate_backend
    normalize = pipeline.normalize
    dedup = pipeline.dedup
    slate_no_opt = pipeline.slate_no_opt

    # -- stream context -----------------------
    stream_ctx = StreamContext()

    # -- load input ---------------------------
    if input_path.lower().endswith(".txt"):
        _run_file_input_text(
            input_path,
            scribe_text_file=scribe_text_file,
            slate_text_file=slate_text_file,
            output_audio_path=output_audio_path,
            slate_audio_path=slate_audio_path,
            input_lang=input_lang,
            text_translation_target=text_translation_target,
            slate_backend=slate_backend,
            normalize=normalize,
            dedup=dedup,
            slate_no_opt=slate_no_opt,
            timers_all=timers_all,
            verbose=verbose,
            stream_ctx=stream_ctx,
        )
    else:
        _run_file_input_audio(
            input_path,
            scribe_text_file=scribe_text_file,
            slate_text_file=slate_text_file,
            output_audio_path=output_audio_path,
            slate_audio_path=slate_audio_path,
            capture_voice_path=capture_voice_path,
            windows=pipeline.window_seconds,
            overlap=pipeline.overlap_seconds,
            rate=None,
            scribe_vad=pipeline.scribe_vad,
            magnitude_threshold=pipeline.magnitude_threshold,
            input_lang=input_lang,
            text_translation_target=text_translation_target,
            scribe_backend=pipeline.scribe_backend,
            slate_backend=slate_backend,
            voice_backend=pipeline.voice_backend,
            voice_model=pipeline.voice_model,
            voice_lang=pipeline.voice_lang,
            voice_match=pipeline.voice_match,
            lang_prefix=pipeline.lang_prefix,
            normalize=normalize,
            dedup=dedup,
            slate_no_opt=slate_no_opt,
            keep_temp=keep_temp,
            timers_all=timers_all,
            verbose=verbose,
            stream_ctx=stream_ctx,
            mqtt_cfg=None,
        )


# -------  Text-input helper  --------------------------------------------------

def _run_file_input_text(
    input_path,
    scribe_text_file,
    slate_text_file,
    output_audio_path,
    slate_audio_path,
    input_lang,
    text_translation_target,
    slate_backend,
    normalize,
    dedup,
    slate_no_opt,
    timers_all,
    verbose,
    stream_ctx,
):
    from ..text_translator import split_into_sentences, translate_text
    from ..tts import synthesize_tts_pcm

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if normalize:
        text = normalize_text(text)

    # -- Determine if we need English (scribe) output -------------------------
    need_scribe = output_audio_path is not None or scribe_text_file is not None

    # -- Determine if target == input (skip Stage 2) -------------------------
    skip_stage2 = False
    if not slate_no_opt and input_lang and text_translation_target:
        input_base = input_lang.split("-")[0]
        target_base = text_translation_target.split("-")[0]
        if input_base == target_base:
            skip_stage2 = True

    # -- Determine if direct translation is possible (non-EN to non-EN) ------
    use_direct = False
    if not slate_no_opt and input_lang and text_translation_target:
        if input_lang.lower() != "en" and text_translation_target.lower() != "en":
            if not need_scribe:
                use_direct = True

    if use_direct:
        # Direct translation: input_lang -> target (no English pivot)
        sentences = split_into_sentences(text)
        translated = []
        for sentence in sentences:
            if timers_all:
                t0 = time.time()
            result = translate_text(
                sentence,
                source_lang=input_lang,
                target_lang=text_translation_target,
                backend=slate_backend,
                verbose=verbose,
            )
            if timers_all:
                elapsed = time.time() - t0
                print(f"[TIMER] Stage 2 translate: {elapsed:.3f}s")
            translated.append(result)
        slate_text = "\n".join(translated)
    else:
        # Stage 1: translate to English (if input not already English)
        if input_lang and input_lang.lower() != "en":
            sentences = split_into_sentences(text)
            english_parts = []
            for sentence in sentences:
                if timers_all:
                    t0 = time.time()
                result = translate_text(
                    sentence,
                    source_lang=input_lang,
                    target_lang="en",
                    backend=slate_backend,
                    verbose=verbose,
                )
                if timers_all:
                    elapsed = time.time() - t0
                    print(f"[TIMER] Stage 1 translate: {elapsed:.3f}s")
                english_parts.append(result)
            english_text = "\n".join(english_parts)
        else:
            english_text = text

        # Write scribe (English) text
        if scribe_text_file:
            with open(scribe_text_file, "w", encoding="utf-8") as f:
                f.write(english_text.upper() if normalize else english_text)
                f.write("\n")

        # Stage 2: translate English -> target
        if not skip_stage2 and text_translation_target:
            sentences = split_into_sentences(english_text)
            translated = []
            for sentence in sentences:
                if timers_all:
                    t0 = time.time()
                result = translate_text(
                    sentence,
                    source_lang="en",
                    target_lang=text_translation_target,
                    backend=slate_backend,
                    verbose=verbose,
                )
                if timers_all:
                    elapsed = time.time() - t0
                    print(f"[TIMER] Stage 2 translate: {elapsed:.3f}s")
                translated.append(result)
            slate_text = "\n".join(translated)
        else:
            slate_text = text  # use original input text

        # Write slate text
        if slate_text_file:
            with open(slate_text_file, "w", encoding="utf-8") as f:
                f.write(slate_text)
                f.write("\n")

    # -- Scribe TTS -----------------------------------------------------------------------
    if output_audio_path:
        tts_lang = "en"
        if stream_ctx.scribe_tts_segments is None:
            stream_ctx.scribe_tts_segments = []
        pcm = synthesize_tts_pcm(
            english_text,
            16000,
            tts_lang,
        )
        if pcm is not None:
            stream_ctx.scribe_tts_segments.append(pcm)

    # -- Slate TTS ------------------------------------------------------------------------
    if slate_audio_path:
        tts_lang = text_translation_target if text_translation_target else "en"
        if stream_ctx.slate_tts_segments is None:
            stream_ctx.slate_tts_segments = []
        pcm = synthesize_tts_pcm(
            slate_text,
            16000,
            tts_lang,
        )
        if pcm is not None:
            stream_ctx.slate_tts_segments.append(pcm)

    # -- Write accumulated audio ----------------------------------------------------------
    if stream_ctx.scribe_tts_segments:
        _write_accumulated_audio(stream_ctx.scribe_tts_segments, output_audio_path)
    if stream_ctx.slate_tts_segments:
        _write_accumulated_audio(stream_ctx.slate_tts_segments, slate_audio_path)


def _write_accumulated_audio(segments, path):
    if not segments:
        return
    audio_data = np.concatenate(segments, axis=0)
    output_audio(audio_data, path)


# -------  Audio-input helper  -------------------------------------------------

def _run_file_input_audio(
    input_path,
    scribe_text_file,
    slate_text_file,
    output_audio_path,
    slate_audio_path,
    capture_voice_path,
    windows,
    overlap,
    rate,
    scribe_vad,
    magnitude_threshold,
    input_lang,
    text_translation_target,
    scribe_backend,
    slate_backend,
    voice_backend,
    voice_model,
    voice_lang,
    voice_match,
    lang_prefix,
    normalize,
    dedup,
    slate_no_opt,
    keep_temp,
    timers_all,
    verbose,
    stream_ctx,
    mqtt_cfg,
):
    from ..stream_output import (
        load_audio_any,
        process_audio_chunk,
    )

    # -- load audio file -----------------------------------------------------------
    audio_data, sample_rate = load_audio_any(input_path, keep_temp=keep_temp)

    if rate is not None:
        sample_rate = rate

    # -- compute window/step -------------------------------------------------------
    frame_count = len(audio_data)

    window_size, hop_size = compute_window_params(
        sample_rate,
        window_seconds=windows,
        overlap_seconds=overlap,
    )

    # -- silence-detection parameters ------------------------------------------
    mag_threshold = magnitude_threshold

    last_scribe_text = None
    last_slate_text = None

    # -- capture-voice accumulator -------------------------------------------------
    capture_chunks = [] if capture_voice_path else None

    # -- process windows -----------------------------------------------------------
    for start_frame in range(0, frame_count, hop_size):
        end_frame = min(start_frame + window_size, frame_count)
        chunk = audio_data[start_frame:end_frame]

        # -- silence gate (when no VAD) ------------------------------------------------
        if not scribe_vad:
            max_abs = np.max(np.abs(chunk))
            if max_abs < mag_threshold:
                continue

        result = process_audio_chunk(
            chunk,
            sample_rate,
            pipeline_cfg=None,
            stream_ctx=stream_ctx,
            mqtt_cfg=mqtt_cfg,
        )

        scribe_text = result.get("scribe")
        slate_text = result.get("slate")

        # -- deduplicate consecutive outputs ------------------------------------------
        if dedup:
            if scribe_text == last_scribe_text:
                scribe_text = None
            if slate_text == last_slate_text:
                slate_text = None
            last_scribe_text = result.get("scribe")
            last_slate_text = result.get("slate")

        # -- write text to files (when produced) ---------------------------------------
        if scribe_text and scribe_text_file:
            with open(scribe_text_file, "a", encoding="utf-8") as f:
                f.write(scribe_text + "\n")
        if slate_text and slate_text_file:
            with open(slate_text_file, "a", encoding="utf-8") as f:
                f.write(slate_text + "\n")

        # -- accumulate capture-voice chunks ------------------------------------------
        if capture_chunks is not None:
            capture_chunks.append(chunk)

    # -- write capture-voice file ---------------------------------------------------
    if capture_chunks:
        capture_audio = np.concatenate(capture_chunks, axis=0)
        output_audio(capture_audio, capture_voice_path)

    # -- write accumulated TTS audio ------------------------------------------------
    if stream_ctx.scribe_tts_segments and output_audio_path:
        _write_accumulated_audio(stream_ctx.scribe_tts_segments, output_audio_path)
    if stream_ctx.slate_tts_segments and slate_audio_path:
        _write_accumulated_audio(stream_ctx.slate_tts_segments, slate_audio_path)