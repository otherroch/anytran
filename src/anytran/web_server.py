import json
import os
import signal

import numpy as np

from .config import get_whisper_backend
from .mqtt_client import init_mqtt
from .processing import process_audio_chunk
from .timing import TimingsAggregator
from .utils import normalize_lang_code, compute_window_params


def run_web_server(
    input_lang=None,
    output_lang=None,
    # output_text_file removed
    magnitude_threshold=0.02,
    model=None,
    verbose=False,
    mqtt_broker=None,
    mqtt_port=1883,
    mqtt_username=None,
    mqtt_password=None,
    mqtt_topic="translation",
    scribe_vad=False,
    host="0.0.0.0",
    port=8000,
    ssl_certfile=None,
    ssl_keyfile=None,
    window_seconds=5.0,
    overlap_seconds=0.0,
    timers=False,
    timers_all=False,
    scribe_backend="auto",
    dedup=False,
    keep_temp=False,
    lang_prefix=False,
    voice_backend=None,
    voice_model=None,
    voice_match=False,
):
    """
    Start the real-time translation web server.

    The server streams audio from a web client, performs speech recognition
    and translation, and can optionally synthesize spoken output using
    Piper TTS. It exposes a WebSocket interface for streaming audio and
    returning partial and final transcription/translation results.

    Parameters
    ----------
    input_lang : str or None, optional
        Default input language code for recognition. If ``None`` or
        ``"auto"``, the server will attempt to automatically detect the
        spoken language.
    output_lang : str or None, optional
        Default output language code for translation. If ``None``, a
        reasonable default such as English is used.

    Returns
    -------
    None
        This function runs the web server and typically does not return
        until the process is terminated.

    Notes
    -----
    Piper TTS configuration is controlled via environment variables:

    ``USE_PIPER``
        When set to ``"1"``, ``"true"``, or ``"yes"`` (case-insensitive),
        enables Piper text-to-speech for generating audio from translated
        text. Any other value (or when unset) disables Piper, and only text
        output will be produced.

    ``PIPER_VOICE``
        Optional name or identifier of the Piper voice to use when
        ``USE_PIPER`` is enabled. This should correspond to a voice
        available in the local Piper installation (for example, a voice
        name or model file base name). See the Piper documentation for the
        list of supported voices and installation instructions.
    """
    # Determine TTS backend: prefer explicit parameter, fall back to env vars.
    if voice_backend is None:
        _use_piper = os.environ.get("USE_PIPER", "0").lower() in ("1", "true", "yes")
        voice_backend = "piper" if _use_piper else "gtts"
    if voice_model is None:
        voice_model = os.environ.get("PIPER_VOICE", None)
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
    except Exception:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        raise

    app = FastAPI()

    default_input_lang = normalize_lang_code(input_lang) if input_lang else "auto"
    default_output_lang = normalize_lang_code(output_lang) if output_lang else "en"
    web_lang_map = {
        "en": "English",
        "fr": "French",
        "zh": "Chinese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
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
    web_lang_items = [("auto", "Auto Detect")] + sorted(web_lang_map.items(), key=lambda item: item[1])
    web_input_lang_options = "\n".join(
        f"<option value=\"{code}\"{' selected' if code == default_input_lang else ''}>{label}</option>"
        for code, label in web_lang_items
    )
    web_output_lang_items = sorted(web_lang_map.items(), key=lambda item: item[1])
    web_output_lang_options = "\n".join(
        f"<option value=\"{code}\"{' selected' if code == default_output_lang else ''}>{label}</option>"
        for code, label in web_output_lang_items
    )

    html_page = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Realtime Audio Translation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; }
        .card { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        button { padding: 10px 16px; margin-right: 8px; }
        label { margin-right: 8px; }
        select { padding: 6px 10px; margin-right: 8px; }
        #log { margin-top: 12px; white-space: pre-wrap; font-family: Consolas, monospace; }
    </style>
</head>
<body>
    <div class=\"card\">
        <h2>Realtime Audio Translation</h2>
        <p>Click Start to stream your microphone audio for translation. The translated text will appear below and be spoken by your browser.</p>
        <button id=\"start\">Start</button>
        <button id=\"stop\" disabled>Stop</button>
        <button id=\"enableAudio\">Enable Audio</button>
        <div style=\"margin-top: 10px;\">
            <label for=\"inputLang\">Input language</label>
            <select id=\"inputLang\">__INPUT_LANG_OPTIONS__</select>
        </div>
        <div style=\"margin-top: 10px;\">
            <label for=\"outputLang\">Output language</label>
            <select id=\"outputLang\">__OUTPUT_LANG_OPTIONS__</select>
        </div>
        <div style=\"margin-top: 10px;\">
            <label for=\"langSwap\">
                <input type=\"checkbox\" id=\"langSwap\" disabled>
                LangSwap (Bidirectional Translation)
            </label>
            <div style=\"margin-left: 20px; font-size: 0.9em; color: #666;\">
                Automatically detect which language is spoken and translate to the other language
            </div>
        </div>
        <div id=\"status\">Idle</div>
        <div id=\"log\"></div>
    </div>

    <script>
        let ws = null;
        let audioContext = null;
        let processor = null;
        let source = null;
        let stream = null;
        let audioEnabled = false;

        const logEl = document.getElementById('log');
        const statusEl = document.getElementById('status');
        const startBtn = document.getElementById('start');
        const stopBtn = document.getElementById('stop');
        const enableAudioBtn = document.getElementById('enableAudio');
        const inputLangSelect = document.getElementById('inputLang');
        const outputLangSelect = document.getElementById('outputLang');
        const langSwapCheckbox = document.getElementById('langSwap');

        function updateLangSwapState() {
            const inputLang = inputLangSelect.value;
            const outputLang = outputLangSelect.value;
            const canEnable = inputLang !== 'auto' && outputLang !== 'auto' && inputLang && outputLang;
            langSwapCheckbox.disabled = !canEnable;
            if (!canEnable) {
                langSwapCheckbox.checked = false;
            }
            sendLangSwapState();
        }

        function sendLangSwapState() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            const payload = { type: 'config', langswap: langSwapCheckbox.checked };
            ws.send(JSON.stringify(payload));
        }

        function logLine(text) {
            logEl.textContent += text + "\\n";
            logEl.scrollTop = logEl.scrollHeight;
        }

        function downsampleTo16k(buffer, inputSampleRate) {
            const outputSampleRate = 16000;
            if (inputSampleRate === outputSampleRate) return buffer;
            const ratio = inputSampleRate / outputSampleRate;
            const newLength = Math.round(buffer.length / ratio);
            const result = new Float32Array(newLength);
            let offsetResult = 0;
            let offsetBuffer = 0;
            while (offsetResult < result.length) {
                const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
                let sum = 0;
                let count = 0;
                for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                    sum += buffer[i];
                    count++;
                }
                result[offsetResult] = count ? (sum / count) : 0;
                offsetResult++;
                offsetBuffer = nextOffsetBuffer;
            }
            return result;
        }

        function floatTo16BitPCM(float32Array) {
            const output = new Int16Array(float32Array.length);
            for (let i = 0; i < float32Array.length; i++) {
                let s = Math.max(-1, Math.min(1, float32Array[i]));
                output[i] = s < 0 ? s * 32768 : s * 32767;
            }
            return output;
        }

        function speak(text, lang = 'en-US') {
            if (!audioEnabled || !('speechSynthesis' in window)) return;
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = lang;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }

        function playPcmAudio(arrayBuffer) {
            if (!audioEnabled) return;
            try {
                const playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
                const int16 = new Int16Array(arrayBuffer);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) {
                    float32[i] = int16[i] / 32768;
                }
                const buffer = playbackCtx.createBuffer(1, float32.length, 16000);
                buffer.getChannelData(0).set(float32);
                const src = playbackCtx.createBufferSource();
                src.buffer = buffer;
                src.connect(playbackCtx.destination);
                src.onended = () => playbackCtx.close();
                src.start();
            } catch (err) {
                logLine('Audio playback error: ' + (err && err.message ? err.message : err));
            }
        }

        function enableAudio() {
            if (!('speechSynthesis' in window)) {
                logLine('Speech synthesis not available in this browser.');
                return;
            }
            try {
                const unlock = new SpeechSynthesisUtterance(' ');
                unlock.volume = 0;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(unlock);
                audioEnabled = true;
                logLine('[info] Audio enabled.');
            } catch (err) {
                logLine('Audio enable failed: ' + (err && err.message ? err.message : err));
            }
        }

        async function start() {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusEl.textContent = 'Connecting...';

            try {
                const wsScheme = location.protocol === 'https:' ? 'wss' : 'ws';
                ws = new WebSocket(`${wsScheme}://${location.host}/ws`);
                ws.binaryType = 'arraybuffer';

                ws.onopen = async () => {
                    statusEl.textContent = 'Requesting mic...';
                    sendInputLang();
                    sendOutputLang();
                    sendLangSwapState();
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    } catch (err) {
                        statusEl.textContent = 'Mic error';
                        logLine('Mic error: ' + (err && err.message ? err.message : err));
                        stop();
                        return;
                    }

                    statusEl.textContent = 'Streaming';
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    source = audioContext.createMediaStreamSource(stream);
                    processor = audioContext.createScriptProcessor(4096, 1, 1);
                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    processor.onaudioprocess = (event) => {
                        if (!ws || ws.readyState !== WebSocket.OPEN) return;
                        const input = event.inputBuffer.getChannelData(0);
                        const downsampled = downsampleTo16k(input, audioContext.sampleRate);
                        const pcm16 = floatTo16BitPCM(downsampled);
                        ws.send(pcm16.buffer);
                    };
                };

                ws.onmessage = (event) => {
                    if (event.data instanceof ArrayBuffer) {
                        playPcmAudio(event.data);
                        return;
                    }
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'translation') {
                            logLine(msg.text);
                            if (!msg.has_audio) {
                                speak(msg.text, msg.lang || 'en-US');
                            }
                        } else if (msg.type === 'info') {
                            logLine('[info] ' + msg.text);
                        }
                    } catch {
                        logLine(event.data);
                    }
                };

                ws.onerror = () => {
                    statusEl.textContent = 'WebSocket error';
                    logLine('WebSocket error');
                };

                ws.onclose = () => {
                    statusEl.textContent = 'Disconnected';
                    stop();
                };
            } catch (err) {
                statusEl.textContent = 'Start error';
                logLine('Start error: ' + (err && err.message ? err.message : err));
                stop();
            }
        }

        function sendInputLang() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            const payload = { type: 'config', input_lang: inputLangSelect.value };
            ws.send(JSON.stringify(payload));
            updateLangSwapState();
        }

        function sendOutputLang() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            const payload = { type: 'config', output_lang: outputLangSelect.value };
            ws.send(JSON.stringify(payload));
            updateLangSwapState();
        }

        function stop() {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusEl.textContent = 'Idle';
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            if (source) {
                source.disconnect();
                source = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (stream) {
                stream.getTracks().forEach(t => t.stop());
                stream = null;
            }
            if (ws) {
                ws.close();
                ws = null;
            }
        }

        startBtn.addEventListener('click', start);
        stopBtn.addEventListener('click', stop);
        enableAudioBtn.addEventListener('click', enableAudio);
        inputLangSelect.addEventListener('change', sendInputLang);
        outputLangSelect.addEventListener('change', sendOutputLang);
        langSwapCheckbox.addEventListener('change', sendLangSwapState);
        
        // Initialize langSwap state on page load
        updateLangSwapState();
    </script>
</body>
</html>
"""
    html_page = html_page.replace("__INPUT_LANG_OPTIONS__", web_input_lang_options)
    html_page = html_page.replace("__OUTPUT_LANG_OPTIONS__", web_output_lang_options)

    @app.get("/")
    def index():
        return HTMLResponse(html_page)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        nonlocal timers
        await websocket.accept()
        text_file = None
        buffer = np.array([], dtype=np.float32)
        rate = 16000
        chunk, overlap = compute_window_params(window_seconds, overlap_seconds, rate)
        current_input_lang = normalize_lang_code(input_lang) if input_lang else None
        current_output_lang = normalize_lang_code(output_lang) if output_lang else "en"
        langswap_enabled = False
        if timers_all: 
            timers = True
        timing_stats = TimingsAggregator("web") if timers else None

        def normalize_web_input_lang(value):
            if value is None:
                return None
            normalized = normalize_lang_code(value)
            if not normalized or normalized == "auto":
                return None
            return normalized

        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] is not None:
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and payload.get("type") == "config":
                        if "input_lang" in payload:
                            current_input_lang = normalize_web_input_lang(payload.get("input_lang"))
                            await websocket.send_text(
                                json.dumps({"type": "info", "text": f"input_lang set to {current_input_lang or 'auto'}"})
                            )
                        if "output_lang" in payload:
                            current_output_lang = normalize_lang_code(payload.get("output_lang")) or "en"
                            await websocket.send_text(
                                json.dumps({"type": "info", "text": f"output_lang set to {current_output_lang}"})
                            )
                        if "langswap" in payload:
                            langswap_enabled = bool(payload.get("langswap"))
                            status = "enabled" if langswap_enabled else "disabled"
                            await websocket.send_text(
                                json.dumps({"type": "info", "text": f"LangSwap {status}"})
                            )
                    continue
                if "bytes" not in message or message["bytes"] is None:
                    continue
                data = message["bytes"]
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, audio_np])
                if len(buffer) >= chunk:
                    audio_segment = buffer[:chunk]
                    buffer = buffer[chunk - overlap :]
                    slate_tts_segments = []
                    translated_text = process_audio_chunk(
                        audio_segment,
                        rate,
                        current_input_lang,
                        current_output_lang,
                        magnitude_threshold,
                        model,
                        verbose,
                        mqtt_broker,
                        mqtt_port,
                        mqtt_username,
                        mqtt_password,
                        mqtt_topic,
                        stream_id="web",
                        scribe_vad=scribe_vad,
                        voice_backend=voice_backend,
                        voice_model=voice_model,
                        timers=timers,
                        timing_stats=timing_stats,
                        scribe_backend=scribe_backend,
                        text_translation_target=current_output_lang,
                        langswap_enabled=langswap_enabled,
                        langswap_input_lang=current_input_lang,
                        langswap_output_lang=current_output_lang,
                        slate_tts_segments=slate_tts_segments,
                        voice_match=voice_match,
                    )
                    if translated_text:
                        # process_audio_chunk returns a dict with 'output', 'scribe', 'slate', and 'final_lang' keys
                        output_text = translated_text.get('output') if isinstance(translated_text, dict) else translated_text
                        final_lang = translated_text.get('final_lang') if isinstance(translated_text, dict) else current_output_lang
                        
                        if output_text:
                            # Use the actual final language from the backend (accounts for LangSwap changes)
                            # Fallback to current_output_lang if not provided
                            lang_code = final_lang or current_output_lang or "en"
                            
                            # Convert language code to IETF language tag (e.g., 'fr' -> 'fr-FR', 'zh-cn' -> 'zh-CN')
                            lang_parts = lang_code.split('-')
                            if len(lang_parts) == 2:
                                web_lang = f"{lang_parts[0]}-{lang_parts[1].upper()}"
                            else:
                                # Common language to locale mappings for Web Speech API
                                lang_map = {
                                    "en": "en-US",
                                    "fr": "fr-FR",
                                    "zh": "zh-CN",
                                    "es": "es-ES",
                                    "de": "de-DE",
                                    "it": "it-IT",
                                    "ja": "ja-JP",
                                    "ko": "ko-KR",
                                    "ru": "ru-RU",
                                    "pt": "pt-BR",
                                    "ar": "ar-SA",
                                    "hi": "hi-IN",
                                    "vi": "vi-VN",
                                    "th": "th-TH",
                                    "tr": "tr-TR",
                                    "nl": "nl-NL",
                                    "sv": "sv-SE",
                                    "no": "no-NO",
                                    "da": "da-DK",
                                    "fi": "fi-FI",
                                    "pl": "pl-PL",
                                    "cs": "cs-CZ",
                                    "el": "el-GR",
                                    "he": "he-IL",
                                    "id": "id-ID",
                                    "ms": "ms-MY",
                                    "uk": "uk-UA",
                                }
                                web_lang = lang_map.get(lang_code, "en-US")
                            
                            if verbose:
                                print(f"[web] Sending translation with language: {lang_code} → {web_lang}")
                            
                            has_tts_audio = any(
                                pcm is not None and len(pcm) > 0 for pcm in slate_tts_segments
                            )
                            message = json.dumps({"type": "translation", "text": output_text, "lang": web_lang, "has_audio": has_tts_audio})
                            await websocket.send_text(message)
                    # Send TTS audio as binary if available
                    for tts_pcm in slate_tts_segments:
                        if tts_pcm is not None and len(tts_pcm) > 0:
                            await websocket.send_bytes(tts_pcm.astype(np.int16).tobytes())
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await websocket.send_text(json.dumps({"type": "info", "text": f"error: {exc}"}))
            except Exception:
                pass
        finally:
            # text_file removed
            if timing_stats is not None:
                backend = get_whisper_backend()
                if backend == "faster-whisper":
                    backend_label = "faster_whisper"
                elif backend == "whisper-ctranslate2":
                    backend_label = "whisper_ctranslate2"
                else:
                    backend_label = "whispercpp"
                if timers_all:
                    summary = timing_stats.format_summary()
                    if summary:
                        print(f"Timing summary (web): {summary}")
                    stage_summary = timing_stats.format_stage_summary()
                    if stage_summary:
                        print(f"Timing summary by stage (web): {stage_summary}")
                    overhead = timing_stats.format_translate_overhead(backend_label)
                    if overhead:
                        print(f"Timing breakdown (web): {overhead}")
                elif timers:
                    stage_summary = timing_stats.format_stage_summary()
                    if stage_summary:
                        print(f"Timing summary by stage (web): {stage_summary}")
    if mqtt_broker:
        init_mqtt(mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic)

    try:
        import uvicorn
    except Exception:
        print("Uvicorn not installed. Install with: pip install uvicorn")
        raise

    use_ssl = bool(ssl_certfile and ssl_keyfile)
    scheme = "https" if use_ssl else "http"
    print(f"Starting web server on {scheme}://{host}:{port}")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
    server = uvicorn.Server(config)

    def signal_handler(sig, frame):
        print("\nStopping web server...", flush=True)
        server.should_exit = True
        server.force_exit = True

    def install_windows_ctrl_handler():
        if os.name != "nt":
            return
        try:
            import ctypes

            def _handler(ctrl_type):
                if ctrl_type in (0, 1):
                    signal_handler(signal.SIGINT, None)
                    return True
                return False

            kernel32 = ctypes.windll.kernel32
            handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
            kernel32.SetConsoleCtrlHandler(handler_type(_handler), True)
        except Exception:
            pass

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    install_windows_ctrl_handler()

    import asyncio

    def suppress_windows_socket_errors(loop, context):
        """
        Custom exception handler to suppress Windows-specific socket errors.
        
        On Windows, when a WebSocket connection is closed by the remote host,
        the ProactorEventLoop may try to call shutdown() on an already closed socket,
        resulting in a ConnectionResetError. This is a known issue and can be safely
        suppressed as it doesn't affect the application's functionality.
        
        See: https://github.com/encode/uvicorn/issues/1316
        """
        exception = context.get('exception')
        if isinstance(exception, ConnectionResetError):
            # Suppress ConnectionResetError on Windows (WinError 10054)
            return
        # For all other exceptions, use the default handler
        loop.default_exception_handler(context)

    async def serve_with_exception_handler():
        """Wrapper to set exception handler for Windows compatibility."""
        if os.name == 'nt':
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(suppress_windows_socket_errors)
        await server.serve()

    try:
        asyncio.run(serve_with_exception_handler())
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
