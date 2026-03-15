from anytran.audio_io import load_audio_any, output_audio
from anytran.processing import build_output_prefix, process_audio_chunk
from anytran.text_translator import translate_text, get_translategemma_model, get_metanllb_model
from anytran.mqtt_client import init_mqtt, send_mqtt_text
from anytran.normalizer import normalize_text, split_into_sentences
from anytran.timing import TimingsAggregator, add_timing
from anytran.tts import play_output, synthesize_tts_pcm
from anytran.utils import compute_window_params
import threading
import numpy as np
import signal
import time
import os
import librosa
 
def run_file_input(
    input_path,
    input_lang=None,
    output_lang=None,
    # output_text_file removed
    magnitude_threshold=0.02,
    # play_audio removed
    output_audio_path=None,
    slate_audio_path=None,
    model=None,
    verbose=False,
    mqtt_broker=None,
    mqtt_port=1883,
    mqtt_username=None,
    mqtt_password=None,
    mqtt_topic="translation",
    scribe_vad=False,
    voice_backend="gtts",
    voice_model=None,
    window_seconds=5.0,
    overlap_seconds=0.0,
    timers=False,
    timers_all=False,
    scribe_backend="auto",
    text_translation_target=None,
    slate_backend="googletrans",
    voice_lang=None,
    scribe_text_file=None,
    slate_text_file=None,
    voice_match=False,
    keep_temp=False,
    dedup=False,
    lang_prefix=False,
    batch=0,
    normalize=True,
    slate_no_opt=False,
):
    print("Starting file input processing...")
    if keep_temp:
        import builtins
        builtins.__dict__["KEEP_TEMP"] = True
    print(f"Input file: {input_path}")
    print(f"Input language: {input_lang}, Output language: {output_lang}")
    if output_audio_path:
        print(f"Scribe audio (English) will be saved to: {output_audio_path}")
    if slate_audio_path:
        print(f"Slate audio (translated) will be saved to: {slate_audio_path}")
    # output_text_file removed
    if scribe_text_file:
        print(f"Scribe text (English) will be saved to: {scribe_text_file}")
    if slate_text_file:
        print(f"Slate text (translated) will be saved to: {slate_text_file}")
    if mqtt_broker:
        print(f"MQTT output enabled: {mqtt_broker}:{mqtt_port}, topic: {mqtt_topic}")

    if mqtt_broker:
        init_mqtt(mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic)

    # Open file handles for all output files
    # output_text_file removed
    scribe_file = open(scribe_text_file, mode="w", encoding="utf-8") if scribe_text_file else None
    slate_file = open(slate_text_file, mode="w", encoding="utf-8") if slate_text_file else None

    # Deduplication tracking for dual text output
    last_scribe_output = None
    last_slate_output = None

    timings = None
    timing_stats = None
    if timers_all:
        # Ensure that enabling all timers also enables the basic timing flag
        timers = True
    if timers:
        print("Timing enabled: Detailed stage timings will be printed after processing.")
        timings = []
        add_timing(timings, "total_start", time.perf_counter()) 
        timing_stats = TimingsAggregator("file") 
        # timing_stats.add(timings)
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".txt":
        # if verbose:
        # print("Processing text file input...")
        if timings:
          t0 = time.perf_counter()
        try:
            with open(input_path, mode="r", encoding="utf-8") as infile:
                text = infile.read()
        except Exception as exc:
            print(f"Error reading input text file: {exc}")
            return
        if timings:
          add_timing(timings, "read_file", t0)

        if not text or not text.strip():
            print("Error: Input text file is empty.")
            return

        english_text = text


        if input_lang and input_lang.lower().split("-")[0] != "en":
            if verbose:
                print("Translating input text to English for Stage 1...split into sentences for better translation quality.")       
            # Chunk the text into sentences for more reliable translation
            if timings:
               t1 = time.perf_counter()            
            sentences = split_into_sentences(text)
            translated_sentences = []
            if timings:
                add_timing(timings, "split_text", t1)                  
                t2 = time.perf_counter() 
            if verbose:
                print("batch size:", batch)
            if slate_backend == "translategemma": 
                get_translategemma_model()  # Preload model before timing translation
            #elif slate_backend == "metanllb":
            #    get_metanllb_model()  # Preload model and tokenizer before timing translation
            if batch > 0:
                    num_sentences = len(sentences)
                    for i in range(0, num_sentences, batch):
                        batch_group = sentences[i:i+batch]
                        batch_start = i + 1
                        batch_end = i + len(batch_group)
                        batch_text = ' '.join(batch_group)
                        if verbose:
                            print(f"Batching sentences {batch_start}-{batch_end} for translation, total length: {len(batch_text)}")
                        translated = translate_text(
                            batch_text,
                            source_lang=input_lang,
                            target_lang="en",
                            backend=slate_backend,
                            verbose=verbose,
                        )
                        if not translated:
                            print(f"Error: Failed to batch translate sentences {batch_start}-{batch_end}.")
                            translated_sentences.extend(batch_group)
                        else:
                            # Split translated batch into sentences
                            batch_translated = split_into_sentences(translated)
                            batch_translated_sentence_count = len(batch_translated) 
                            if verbose:
                                print(f"Batch translated text length: {len(translated)}")
                                print(f"Batch translated num sentences before splitting: {batch_translated_sentence_count}")    
                            if batch_translated_sentence_count == 0:
                                if translated.strip():
                                    translated_sentences.append(translated.strip())
                            else:
                                for sentence in batch_translated:
                                    if sentence.strip():
                                        translated_sentences.append(sentence.strip())
                            if verbose:
                                print(f"translated sentences after splitting: {len(translated_sentences)}")
                    english_text = ' '.join(translated_sentences)
                    if verbose:
                        print(f"Total translated text length after batching: {len(english_text)}")
            else:
                for idx, sent in enumerate(sentences):
                    if verbose:
                        print(f"Translating sentence {idx+1}/{len(sentences)}: {sent}")
                    translated = translate_text(
                        sent,
                        source_lang=input_lang,
                        target_lang="en",
                        backend=slate_backend,
                        verbose=verbose,
                    )
                    if not translated:
                        print(f"Error: Failed to translate sentence {idx+1}: {sent}")
                        continue
                    if verbose:
                        print(f"Translated sentence {idx+1}: {translated}") 
                    translated_sentences.append(translated)
                english_text = ' '.join(translated_sentences)
            if verbose:
                print(f"Stage 1 (Text -> English) len: {len(english_text)}")

            if timings:
               add_timing(timings, "stage2_translation", t2)   

        # Write Stage 1 (English) to scribe file, prefixing each sentence
        if scribe_file and english_text:
            if timings:
                t3 = time.perf_counter()
            sentences = split_into_sentences(english_text)
            if verbose:
                print(f"Writing {len(sentences)} sentences to scribe file...")   
            for sent in sentences:
                if lang_prefix: 
                    scribe_output = f"{build_output_prefix(detected_lang='en')}{sent}" 
                else:
                    scribe_output = sent
                if normalize:
                    scribe_output = normalize_text(scribe_output)
                scribe_file.write(f"{scribe_output}\n")
            scribe_file.flush()

            if timings:
                add_timing(timings, "write_scribe_file", t3)

        final_text = english_text
        final_lang = "en"
        stage2_ran = False
        if text_translation_target and text_translation_target.lower().split("-")[0] != "en":
            if timings:
                t6 = time.perf_counter()
            # Optimization: if the input is already in the target language, skip
            # the round-trip translation (input_lang → English → input_lang) and
            # use the original text directly as the slate output.
            # This can be disabled with slate_no_opt=True.
            input_base = input_lang.lower().split("-")[0] if input_lang else None
            target_base = text_translation_target.lower().split("-")[0]
            if not slate_no_opt and input_base and input_base == target_base:
                if verbose:
                    print(
                        f"Input language matches target language ({text_translation_target}), "
                        "skipping Stage 2 translation and using original text directly."
                    )
                if timings:
                    add_timing(timings, "stage2_translation", t6)
                final_text = text
                final_lang = text_translation_target
                stage2_ran = True
            else:
                if verbose:
                    print(f"Translating text from English to {text_translation_target} for Stage 2...")
                if batch > 0:
                    # Split English text into sentences for Stage 2 batching
                    sentences = split_into_sentences(english_text)
                    translated_sentences = []
                    for i in range(0, len(sentences), batch):
                        batch_group = sentences[i:i+batch]
                        batch_text = ' '.join(batch_group)
                        if verbose:
                            print(f"Batching sentences {i+1}-{i+len(batch_group)} for translation, total length: {len(batch_text)}")
                        translated = translate_text(
                            batch_text,
                            source_lang="en",
                            target_lang=text_translation_target,
                            backend=slate_backend,
                            verbose=verbose,
                        )
                        if not translated:
                            print(f"Error: Failed to batch translate sentences {i+1}-{i+len(batch_group)}.")
                            translated_sentences.extend(batch_group)
                        else:
                            # Split translated batch into sentences
                            batch_translated = split_into_sentences(translated)
                            for sentence in batch_translated:
                                if sentence.strip():
                                    translated_sentences.append(sentence.strip())
                    translated = ' '.join(translated_sentences)
                else:
                    # Split English text into sentences for Stage 2 if not batching
                    sentences = split_into_sentences(english_text)
                    translated_sentences = []
                    for idx, sent in enumerate(sentences):
                        translated_sent = translate_text(
                            sent,
                            source_lang="en",
                            target_lang=text_translation_target,
                            backend=slate_backend,
                            verbose=verbose,
                        )
                        if not translated_sent:
                            print(f"Error: Failed to translate sentence {idx+1}: {sent}")
                            continue
                        translated_sentences.append(translated_sent)
                    translated = ' '.join(translated_sentences)

                if timings:
                    add_timing(timings, "stage2_translation", t6)

                if translated:
                    final_text = translated
                    final_lang = text_translation_target
                    stage2_ran = True
                    if verbose:
                        print(f"Stage 2 (English -> {text_translation_target}) len: {len(final_text)}")

        
        # Write to slate file if provided, but avoid duplicating unchanged English text
        if slate_file and final_text:
            if verbose:
                print(f"Writing translated text to slate file...")
            if timings:
                t8 = time.perf_counter()
            if lang_prefix:
                slate_output = f"{build_output_prefix(detected_lang=final_lang)}{final_text.strip()}"
            else:
                slate_output = final_text.strip()
            if normalize:
                slate_output = normalize_text(slate_output)
            slate_file.write(f"{slate_output}\n")
            slate_file.flush()
            if timings:
               add_timing(timings, "write_slate_file", t8)

        if not final_text:
            print("Error: No output text generated from input file.")
            return

        if lang_prefix:  
           output_text = f"{build_output_prefix(detected_lang=final_lang)}{final_text.strip()}" 
        else: 
            output_text = final_text.strip()
        # output_text_file removed

        if mqtt_broker:
            send_mqtt_text(output_text, mqtt_topic, mqtt_broker, mqtt_port, mqtt_username, mqtt_password)

        # Stage 1 audio (English scribe voice)
        if output_audio_path and english_text:
            if timings:
                t12 = time.perf_counter()   
            tts_lang = voice_lang or "en"
            tts_pcm = synthesize_tts_pcm(
                english_text,
                16000,
                tts_lang,
                voice_backend=voice_backend,
                voice_model=voice_model,
                verbose=verbose,
            )
            if tts_pcm is not None:
                try:
                    output_audio(tts_pcm, output_audio_path, play=False)
                    print(f"Scribe audio file saved: {output_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving scribe audio file: {exc}", flush=True)
            if timings:
                add_timing(timings, "stage3_synthesis", t12)  

        # Stage 2 audio (translated slate voice) - only if translation occurred
        if slate_audio_path and stage2_ran and final_text:
            if timings:
                t14 = time.perf_counter()   
            tts_lang = voice_lang or text_translation_target or output_lang or "en"
            tts_pcm = synthesize_tts_pcm(
                final_text,
                16000,
                tts_lang,
                voice_backend=voice_backend,
                voice_model=voice_model,
                verbose=verbose,
            )
            if tts_pcm is not None:
                try:
                    output_audio(tts_pcm, slate_audio_path, play=False)
                    print(f"Slate audio file saved: {slate_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving slate audio file: {exc}", flush=True)
            if timings:
                add_timing(timings, "stage3_synthesis", t14)

        # play_audio removed
        if timings:
            timing_stats.add(timings, "chunk")
            if timers_all:
                summary = timing_stats.format_summary()
                if summary:
                    print(f"\nTiming summary (file):\n{summary}")
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (file):\n{stage_summary}")
                overhead = timing_stats.format_translate_overhead('chunk')
                if overhead:
                    print(f"\nTiming translate overhead (file):\n{overhead}")
            elif timings:
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (file):\n{stage_summary}")
                    
        if scribe_file: 
            scribe_file.close()
        if slate_file:
            slate_file.close()
            
        return # Exit after processing text file, since we don't have audio to chunk/process

    try:
        audio, rate = load_audio_any(input_path)
    except Exception as exc:
        print(f"Error loading input audio file: {exc}")
        return

    if audio is None:
        print("Error: Failed to load audio data from input file.")
        return

    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if not np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    if rate != 16000:
        try:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            if verbose:
                print(f"Resampled audio from {rate} Hz to 16000 Hz")
            rate = 16000
        except Exception as exc:
            print(f"Error resampling audio: {exc}")
            return

    audio_segments = [] if output_audio_path else None
    slate_audio_segments = [] if slate_audio_path else None
    if timings:
        ta0 = time.perf_counter()
    chunk, overlap = compute_window_params(window_seconds, overlap_seconds, rate)
    step = max(1, chunk - overlap)
    try:
        for start in range(0, len(audio), step):
            audio_segment = audio[start : start + chunk]
            if audio_segment.size == 0:
                break
            result = process_audio_chunk(
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
                stream_id="file",
                scribe_vad=scribe_vad,
                voice_backend=voice_backend,
                voice_model=voice_model,
                timers=timers,
                timing_stats=timing_stats,
                scribe_backend=scribe_backend,
                text_translation_target=text_translation_target,
                slate_backend=slate_backend,
                voice_lang=voice_lang,
                scribe_text_file=None,
                slate_text_file=None,
                scribe_tts_segments=audio_segments,
                slate_tts_segments=slate_audio_segments,
                voice_match=voice_match,
                lang_prefix=lang_prefix,
            )

            # Deduplication: Write outputs only if different from last ones
            if result:
                scribe_output = result.get('scribe')
                slate_output = result.get('slate')

                if scribe_output and scribe_output != last_scribe_output:
                    if scribe_file:
                        if normalize:
                            scribe_output = normalize_text(scribe_output)
                        scribe_file.write(f"{scribe_output}\n")
                        scribe_file.flush()
                    last_scribe_output = scribe_output

                if slate_output and slate_output != last_slate_output:
                    if slate_file:
                        if normalize:
                            slate_output = normalize_text(slate_output)
                        slate_file.write(f"{slate_output}\n")
                        slate_file.flush()
                    last_slate_output = slate_output
    finally:
        # output_text_file removed
        if scribe_file:
            scribe_file.close()
            print(f"Scribe text file saved: {scribe_text_file}", flush=True)
        if slate_file:
            slate_file.close()
            print(f"Slate text file saved: {slate_text_file}", flush=True)
        if audio_segments is not None:
            if len(audio_segments) > 0:
                all_audio = np.concatenate(audio_segments)
                try:
                    output_audio(all_audio, output_audio_path, play=False)
                    print(f"Output audio file saved: {output_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving output audio file: {exc}", flush=True)
        if slate_audio_segments is not None:
            if len(slate_audio_segments) > 0:
                all_slate_audio = np.concatenate(slate_audio_segments)
                try:
                    output_audio(all_slate_audio, slate_audio_path, play=False)
                    print(f"Slate audio file saved: {slate_audio_path}", flush=True)
                except Exception as exc:
                    print(f"Error saving slate audio file: {exc}", flush=True)
        if timings:
            add_timing(timings, "stage1_transcription", ta0)
            timing_stats.add(timings, "chunk")
            if timers_all:
                summary = timing_stats.format_summary()
                if summary:
                    print(f"\nTiming summary (file):\n{summary}")
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (file):\n{stage_summary}")
                overhead = timing_stats.format_translate_overhead('chunk')
                if overhead:
                    print(f"\nTiming translate overhead (file):\n{overhead}")
            elif timings:
                stage_summary = timing_stats.format_stage_summary()
                if stage_summary:
                    print(f"\nTiming summary by stage (file):\n{stage_summary}")
