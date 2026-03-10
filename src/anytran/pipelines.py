import itertools
import os
import tempfile

from .normalizer import normalize_text
from .runners import run_file_input, run_multi_rtsp, run_realtime_output, run_realtime_rtsp, run_realtime_youtube
from .web_server import run_web_server


def build_pipeline_config(args):
    """Build pipeline configuration from arguments."""
    # Normalize input_lang (None means auto)
    input_lang = args.input_lang
    
    # Determine if we need translation (Stage 2)
    # Allow slate options even when output_lang is English
    needs_translation = args.output_lang.lower() != "en"
    output_lang_for_tts = args.output_lang if needs_translation else "en"
    
    config = {
        # Input
        "input_lang": input_lang,
        "output_lang": args.output_lang,
        
        # Stage 1 outputs (English transcription)
        "scribe_text": args.scribe_text,
        "scribe_voice": args.scribe_voice,
        
        # Stage 2 outputs (Translation or re-output of Stage 1)
        # Now allow slate options even when output_lang is English
        "needs_translation": needs_translation,
        "slate_text": args.slate_text,
        "slate_voice": args.slate_voice,
        
        # text_output removed (no longer needed)
        
        # Backend
        "model": args.scribe_model,
        "scribe_backend": args.scribe_backend,
        "magnitude_threshold": args.magnitude_threshold,
        
        # Translation
        "text_translation_target": args.output_lang if needs_translation else None,
        "slate_backend": args.slate_backend,
        
        # TTS
        "voice_lang": args.voice_lang or output_lang_for_tts,
        "voice_backend": args.voice_backend,
        "voice_model": args.voice_model,
        "voice_match": args.voice_match,
        
        # Audio processing
        "scribe_vad": args.scribe_vad,
        "window_seconds": args.window_seconds,
        "overlap_seconds": args.overlap_seconds,
        
        # MQTT
        "mqtt_broker": args.mqtt_broker,
        "mqtt_port": args.mqtt_port,
        "mqtt_username": args.mqtt_username,
        "mqtt_password": args.mqtt_password,
        "mqtt_topic": args.mqtt_topic,
        
        # Misc
        "verbose": args.verbose,
        "timers": args.timers,
        "timers_all": getattr(args, "timers_all", False),
        "chat_log_dir": args.chat_log,
        "keep_temp": getattr(args, "keep_temp", False),
        "dedup": getattr(args, "dedup", False),
        "lang_prefix": getattr(args, "lang_prefix", False),
        "normalize": not getattr(args, "no_norm", False),
        "normalize_input": not getattr(args, "no_input_norm", False),

        # Capture original input voice
        "capture_voice": getattr(args, "capture_voice", None),
    }
    
    return config


def execute_pipeline(args, config):
    """Execute the appropriate pipeline based on input source."""
    
    if args.web:
        return _run_web_pipeline(args, config)
    elif args.from_output:
        return _run_output_pipeline(args, config)
    elif args.youtube_url:
        return _run_youtube_pipeline(args, config)
    elif args.rtsp:
        return _run_rtsp_pipeline(args, config)
    elif args.input:
        return _run_file_pipeline(args, config)
    else:
        print("Error: No input source specified")
        return 1


def _run_web_pipeline(args, config):
    """Run web server pipeline."""
    run_web_server(
        config["input_lang"],
        config["output_lang"],
        config["magnitude_threshold"],
        model=config["model"],
        verbose=config["verbose"],
        mqtt_broker=config["mqtt_broker"],
        mqtt_port=config["mqtt_port"],
        mqtt_username=config["mqtt_username"],
        mqtt_password=config["mqtt_password"],
        mqtt_topic=config["mqtt_topic"],
        scribe_vad=config["scribe_vad"],
        host=args.web_host,
        port=args.web_port,
        ssl_certfile=args.web_ssl_cert,
        ssl_keyfile=args.web_ssl_key,
        window_seconds=config["window_seconds"],
        overlap_seconds=config["overlap_seconds"],
        timers=config["timers"],
        timers_all=config["timers_all"],
        scribe_backend=config["scribe_backend"],
        slate_backend=config["slate_backend"],
        dedup=config["dedup"],
        lang_prefix=config["lang_prefix"],
        voice_backend=config["voice_backend"],
        voice_model=config["voice_model"],
        voice_match=config["voice_match"],
        capture_voice_path=config.get("capture_voice"),
    )
    return 0


def _run_output_pipeline(args, config):
    """Run system output capture pipeline."""
    run_realtime_output(
        config["input_lang"],
        config["output_lang"],
        config["magnitude_threshold"],
        output_audio_path=config["scribe_voice"] if isinstance(config["scribe_voice"], str) else None,
        slate_audio_path=config["slate_voice"] if isinstance(config["slate_voice"], str) else None,
        model=config["model"],
        verbose=config["verbose"],
        mqtt_broker=config["mqtt_broker"],
        mqtt_port=config["mqtt_port"],
        mqtt_username=config["mqtt_username"],
        mqtt_password=config["mqtt_password"],
        mqtt_topic=config["mqtt_topic"],
        scribe_vad=config["scribe_vad"],
        voice_backend=config["voice_backend"],
        voice_model=config["voice_model"],
        output_device=args.output_device,
        window_seconds=config["window_seconds"],
        overlap_seconds=config["overlap_seconds"],
        timers=config["timers"],
        timers_all=config["timers_all"],
        scribe_backend=config["scribe_backend"],
        text_translation_target=config["text_translation_target"],
        slate_backend=config["slate_backend"],
        voice_lang=config["voice_lang"],
        scribe_text_file=config["scribe_text"],
        slate_text_file=config["slate_text"],
        voice_match=config["voice_match"],
        dedup=config["dedup"],
        lang_prefix=config["lang_prefix"],
        normalize=config.get("normalize", True),
        capture_voice_path=config.get("capture_voice"),
    )
    return 0


def _run_youtube_pipeline(args, config):
    """Run YouTube streaming pipeline."""
    run_realtime_youtube(
        args.youtube_url,
        args.youtube_api_key,
        config["input_lang"],
        config["output_lang"],
        config["magnitude_threshold"],
        output_audio_path=config["scribe_voice"] if isinstance(config["scribe_voice"], str) else None,
        slate_audio_path=config["slate_voice"] if isinstance(config["slate_voice"], str) else None,
        model=config["model"],
        verbose=config["verbose"],
        mqtt_broker=config["mqtt_broker"],
        mqtt_port=config["mqtt_port"],
        mqtt_username=config["mqtt_username"],
        mqtt_password=config["mqtt_password"],
        mqtt_topic=config["mqtt_topic"],
        scribe_vad=config["scribe_vad"],
        voice_backend=config["voice_backend"],
        voice_model=config["voice_model"],
        youtube_js_runtime=args.youtube_js_runtime,
        youtube_remote_components=args.youtube_remote_components,
        window_seconds=config["window_seconds"],
        overlap_seconds=config["overlap_seconds"],
        timers=config["timers"],
        timers_all=config["timers_all"],
        scribe_backend=config["scribe_backend"],
        text_translation_target=config["text_translation_target"],
        slate_backend=config["slate_backend"],
        voice_lang=config["voice_lang"],
        scribe_text_file=config["scribe_text"],
        slate_text_file=config["slate_text"],
        voice_match=config["voice_match"],
        dedup=config["dedup"],
        lang_prefix=config["lang_prefix"],
        normalize=config.get("normalize", True),
        capture_voice_path=config.get("capture_voice"),
    )
    return 0


def _run_rtsp_pipeline(args, config):
    """Run RTSP streaming pipeline."""
    if len(args.rtsp) == 1:
        topic = args.mqtt_topic_names[0] if args.mqtt_topic_names else config["mqtt_topic"]
        run_realtime_rtsp(
            args.rtsp[0],
            config["input_lang"],
            config["output_lang"],
            config["magnitude_threshold"],
            output_audio_path=config["scribe_voice"] if isinstance(config["scribe_voice"], str) else None,
            slate_audio_path=config["slate_voice"] if isinstance(config["slate_voice"], str) else None,
            model=config["model"],
            verbose=config["verbose"],
            mqtt_broker=config["mqtt_broker"],
            mqtt_port=config["mqtt_port"],
            mqtt_username=config["mqtt_username"],
            mqtt_password=config["mqtt_password"],
            mqtt_topic=topic,
            scribe_vad=config["scribe_vad"],
            voice_backend=config["voice_backend"],
            voice_model=config["voice_model"],
            chat_log_dir=config["chat_log_dir"],
            window_seconds=config["window_seconds"],
            overlap_seconds=config["overlap_seconds"],
            timers=config["timers"],
            timers_all=config["timers_all"],
            scribe_backend=config["scribe_backend"],
            text_translation_target=config["text_translation_target"],
            slate_backend=config["slate_backend"],
            voice_lang=config["voice_lang"],
            scribe_text_file=config["scribe_text"],
            slate_text_file=config["slate_text"],
            voice_match=config["voice_match"],
            dedup=config["dedup"],
            lang_prefix=config["lang_prefix"],
            normalize=config.get("normalize", True),
            capture_voice_path=config.get("capture_voice"),
        )
    else:
        run_multi_rtsp(
            args.rtsp,
            config["input_lang"],
            config["output_lang"],
            config["scribe_voice"] if isinstance(config["scribe_voice"], str) else None,
            config["slate_voice"] if isinstance(config["slate_voice"], str) else None,
            config["magnitude_threshold"],
            model=config["model"],
            verbose=config["verbose"],
            mqtt_broker=config["mqtt_broker"],
            mqtt_port=config["mqtt_port"],
            mqtt_username=config["mqtt_username"],
            mqtt_password=config["mqtt_password"],
            mqtt_topic=config["mqtt_topic"],
            topic_names=args.mqtt_topic_names,
            scribe_vad=config["scribe_vad"],
            voice_backend=config["voice_backend"],
            voice_model=config["voice_model"],
            chat_log_dir=config["chat_log_dir"],
            window_seconds=config["window_seconds"],
            overlap_seconds=config["overlap_seconds"],
            timers=config["timers"],
            timers_all=config["timers_all"],
            scribe_backend=config["scribe_backend"],
            text_translation_target=config["text_translation_target"],
            slate_backend=config["slate_backend"],
            voice_lang=config["voice_lang"],
            scribe_text_file=config["scribe_text"],
            slate_text_file=config["slate_text"],
            voice_match=config["voice_match"],
            dedup=config["dedup"],
            lang_prefix=config["lang_prefix"],
            normalize=config.get("normalize", True),
            capture_voice_path=config.get("capture_voice"),
        )
    return 0


def _files_are_identical(path_a, path_b):
    """Return True if both files exist and have identical content."""
    try:
        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


MAX_DIFF_LINES_TO_DISPLAY = 5

def _files_line_difference(path_a, path_b):
    """Return the number of lines that differ between two text files."""
    try:
        with open(path_a, "r", encoding="utf-8") as fa, open(path_b, "r", encoding="utf-8") as fb:
            return sum(1 for a, b in itertools.zip_longest(fa, fb, fillvalue="") if a != b)
    except OSError:
        return float('inf')

def _files_line_differences_with_lines(path_a, path_b):
    """Return (count, list of (idx, line_a, line_b)) for differing lines between two text files."""
    differences = []
    try:
        with open(path_a, "r", encoding="utf-8") as fa, open(path_b, "r", encoding="utf-8") as fb:
            lines_a = fa.readlines()
            lines_b = fb.readlines()
        max_len = max(len(lines_a), len(lines_b))
        lines_a += [""] * (max_len - len(lines_a))
        lines_b += [""] * (max_len - len(lines_b))
        for idx, (a, b) in enumerate(zip(lines_a, lines_b), 1):
            if a != b:
                differences.append((idx, a.rstrip(), b.rstrip()))
        return len(differences), differences
    except Exception:
        return float('inf'), []


def _run_file_pipeline(args, config):
    """Run file-based pipeline (audio or text input)."""
    
    input_path = args.input
    is_text_input = input_path and input_path.lower().endswith(".txt")
    temp_input_path = input_path
    normalize_input = config.get("normalize", True) and config.get("normalize_input", True)
    verbose = config["verbose"]
    keep_temp = config.get("keep_temp", False)  

    input_is_temp_file = False  
    
    if is_text_input and normalize_input:
        try:
            with open(input_path, "r", encoding="utf-8") as infile:
                original_text = infile.read()
                normalized_text = normalize_text(original_text)
                if normalized_text != original_text:
                    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as tmpfile:
                        tmpfile.write(normalized_text)
                        temp_input_path = tmpfile.name
                    input_is_temp_file = True
                    if verbose:
                        print(f"Input text file normalized. Using temporary file: {temp_input_path}")
                        # Show the differing lines with line numbers
                        small_diff_lines, differences = _files_line_differences_with_lines(temp_input_path, input_path)
                        print(f"Small number of differing lines ({small_diff_lines}) detected. Showing up to {MAX_DIFF_LINES_TO_DISPLAY} differing lines:")
                        for idx, a, b in differences[:MAX_DIFF_LINES_TO_DISPLAY]:
                            print(f"    Line {idx}:")
                            print(f"      {os.path.basename(input_path)}: {b}") 
                            print(f"      {os.path.basename(temp_input_path)}: {a}")

                             
        except Exception as exc:
            print(f"Error normalizing input file: {exc}")
            
    # first run with original input (normalized temp file if normalization was applied) - this will produce the initial slate_text output
    run_file_input(
        temp_input_path,
        config["input_lang"],
        config["output_lang"],
        config["magnitude_threshold"],
        output_audio_path=config["scribe_voice"] if isinstance(config["scribe_voice"], str) else None,
        slate_audio_path=config["slate_voice"] if isinstance(config["slate_voice"], str) else None,
        model=config["model"],
        verbose=config["verbose"],
        mqtt_broker=config["mqtt_broker"],
        mqtt_port=config["mqtt_port"],
        mqtt_username=config["mqtt_username"],
        mqtt_password=config["mqtt_password"],
        mqtt_topic=config["mqtt_topic"],
        scribe_vad=config["scribe_vad"],
        voice_backend=config["voice_backend"],
        voice_model=config["voice_model"],
        window_seconds=config["window_seconds"],
        overlap_seconds=config["overlap_seconds"],
        timers=config["timers"],
        timers_all=config["timers_all"],
        scribe_backend=config["scribe_backend"],
        text_translation_target=config["text_translation_target"],
        slate_backend=config["slate_backend"],
        voice_lang=config["voice_lang"],
        scribe_text_file=config["scribe_text"],
        slate_text_file=config["slate_text"],
        voice_match=config["voice_match"],
        keep_temp=config.get("keep_temp", False),
        dedup=config["dedup"],
        lang_prefix=config["lang_prefix"],
        batch=args.batch_input_text,
        normalize=config.get("normalize", True),
    )

    if input_is_temp_file and not keep_temp:    
        try:
            os.remove(temp_input_path)
            if verbose:
                print(f"Removed temporary normalized input file: {temp_input_path}")
        except OSError as exc:
            print(f"Warning: Unable to remove temporary file '{temp_input_path}': {exc}")
            
    # Handle --looptran: repeat translation with swapped languages for text files
    looptran_n = getattr(args, "looptran", 0)
    slate_text = config["slate_text"]
    input_lang = config["input_lang"]
    output_lang = config["output_lang"]

    tran_converge_flag = False
    tran_converge_diff = 0
    if args.tran_converge is not None:
        tran_converge_diff = args.tran_converge
        tran_converge_flag = True

    if (
        looptran_n > 0
        and is_text_input
        and slate_text
        and input_lang
        and output_lang
        and input_lang.lower() != output_lang.lower()
    ):
        slate_base, slate_ext = os.path.splitext(slate_text)
        current_input_path = slate_text
        current_input_lang = output_lang
        current_output_lang = input_lang

        scribe_text = config.get("scribe_text")
        if scribe_text:
            scribe_base, scribe_ext = os.path.splitext(scribe_text)
        else:
            scribe_base, scribe_ext = None, None

        # Keep track of output slate paths for convergence comparison (index 0 = initial slate_text)
        output_slate_paths = [slate_text]

        for i in range(1, looptran_n + 1):
            new_slate_text = f"{slate_base}_{i}{slate_ext}"
            new_scribe_text = f"{scribe_base}_{i}{scribe_ext}" if scribe_base else None
            needs_translation = current_output_lang.lower() != "en"
            text_translation_target = current_output_lang if needs_translation else None
            voice_lang = args.voice_lang or (current_output_lang if needs_translation else "en")

            run_file_input(
                current_input_path,
                current_input_lang,
                current_output_lang,
                config["magnitude_threshold"],
                output_audio_path=config["scribe_voice"] if isinstance(config["scribe_voice"], str) else None,
                slate_audio_path=config["slate_voice"] if isinstance(config["slate_voice"], str) else None,
                model=config["model"],
                verbose=config["verbose"],
                mqtt_broker=config["mqtt_broker"],
                mqtt_port=config["mqtt_port"],
                mqtt_username=config["mqtt_username"],
                mqtt_password=config["mqtt_password"],
                mqtt_topic=config["mqtt_topic"],
                scribe_vad=config["scribe_vad"],
                voice_backend=config["voice_backend"],
                voice_model=config["voice_model"],
                window_seconds=config["window_seconds"],
                overlap_seconds=config["overlap_seconds"],
                timers=config["timers"],
                timers_all=config["timers_all"],
                scribe_backend=config["scribe_backend"],
                text_translation_target=text_translation_target,
                slate_backend=config["slate_backend"],
                voice_lang=voice_lang,
                scribe_text_file=new_scribe_text,
                slate_text_file=new_slate_text,
                voice_match=config["voice_match"],
                keep_temp=config.get("keep_temp", False),
                dedup=config["dedup"],
                lang_prefix=config["lang_prefix"],
                batch=args.batch_input_text,
                normalize=config.get("normalize", True),
            )

            output_slate_paths.append(new_slate_text)

            # Check for convergence: compare this output with the one two steps back (same language)
            if tran_converge_flag and i >= 2:
                prev_same_lang_path = output_slate_paths[i - 2]
                diff_lines = _files_line_difference(new_slate_text, prev_same_lang_path)
                threshold = tran_converge_diff
                print(
                    f"[tran-converge] Iteration {i}: '{new_slate_text}' vs '{prev_same_lang_path}' differ by {diff_lines} lines (threshold={threshold})."
                )
 
                if diff_lines < MAX_DIFF_LINES_TO_DISPLAY:
                    # Show the differing lines with line numbers
                    small_diff_lines, differences = _files_line_differences_with_lines(new_slate_text, prev_same_lang_path)
                    if small_diff_lines > 0:
                         if small_diff_lines <= MAX_DIFF_LINES_TO_DISPLAY:
                            print(f"small number of differing lines ({small_diff_lines}) detected :")
                         else:
                            print(f"large number of differing lines ({small_diff_lines}) detected. Showing up to {MAX_DIFF_LINES_TO_DISPLAY} differing lines:")
                         for idx, a, b in differences:
                            print(f"    Line {idx}:")
                            print(f"      {os.path.basename(new_slate_text)}: {a}")
                            print(f"      {os.path.basename(prev_same_lang_path)}: {b}")
                if diff_lines <= threshold:
                    print(
                        f"[tran-converge] Convergence detected at iteration {i}: "
                        f"'{new_slate_text}' and '{prev_same_lang_path}' differ by {diff_lines} lines (threshold={threshold}). Stopping early."
                    )
                    break

            # For next pass, swap languages and use new files as input
            current_input_lang, current_output_lang = current_output_lang, current_input_lang
            current_input_path = new_slate_text

    return 0
