# Command Line Options Reference

This document describes all available command line options for the anytran audio translator and how they interact with each other.

---

## Architecture Overview

The anytran pipeline has three stages:
1. **Stage 1 (Transcription)**: Audio → English text (using Whisper transcription)
2. **Stage 2 (Translation)**: English text → Output language text (only if `--output-lang` ≠ `en`)
3. **Stage 3 (TTS)**: Text → Audio (if voice output is requested)

---

## Input Sources (Mutually Exclusive)

These options define where audio/text comes from. **Exactly one must be specified**.

### `--input <file>`
- **Type**: File path
- **Supported formats**: `.txt`, `.mp3`, `.mp4`, `.wav`, `.m4a`, `.flac`, `.ogg`
- **Behavior**:
  - For text files: Skips Stage 1 (transcription), starts at Stage 2 (text translation)
  - For audio files: Processes through full pipeline
  - Text file input **requires** `--input-lang` to be specified (not `auto`)
- **Interactions**: 
  - Incompatible with `--rtsp`, `--web`, `--from-output`, `--youtube-url`

### `--rtsp <url>`
- **Type**: RTSP stream URL
- **Can be repeated**: Yes (for multiple streams: `--rtsp url1 --rtsp url2`)
- **Behavior**: Processes streaming RTSP sources in real-time
- **Single vs. Multiple streams**:
  - Single stream: Uses `run_realtime_rtsp()` (optimized for single stream)
  - Multiple streams: Uses `run_multi_rtsp()` with parallel processing
- **Interactions**:
  - `--mqtt-topic-names`: If specified, count must match number of `--rtsp` arguments
  - Requires at least one output option
  - Incompatible with `--input`, `--web`, `--from-output`, `--youtube-url`

### `--web`
- **Type**: Boolean flag
- **Behavior**: Starts a web server for browser-based microphone input
- **Typical port**: 8443 (HTTPS) — configurable with `--web-port`
- **Certificate handling**:
  - Requires either `--web-ssl-cert` and `--web-ssl-key` OR `--web-ssl-self-signed`
  - Both cert parameters must be provided together or neither
- **Interactions**:
  - Incompatible with `--input`, `--rtsp`, `--from-output`, `--youtube-url`
  - Works with all output options
  - Requires web server config options if non-standard

### `--from-output`
- **Type**: Boolean flag
- **Platform**: Windows only (uses WASAPI loopback capture)
- **Behavior**: Captures system audio output (speaker output) in real-time
- **Device selection**: 
  - Use `--output-device` to select specific loopback device
  - Use `--list-output-devices` to see available options
- **Requires**: At least one text output option (`--scribe-text` or `--slate-text`)
- **Interactions**:
  - Requires text output (audio output won't be used for system capture)
  - Not recommended with `--from-output` and voice output simultaneously
  - Incompatible with `--input`, `--rtsp`, `--web`, `--youtube-url`

### `--youtube-url <url>`
- **Type**: YouTube video URL
- **Requires**: `--youtube-api-key` (YouTube Data API v3 key)
- **Behavior**: Extracts audio from YouTube video and processes it
- **Requires output**: At least one text output option (`--scribe-text` or `--slate-text`)
- **Optional YouTube parameters**:
  - `--youtube-js-runtime`: JS runtime for yt-dlp (default: `node`)
  - `--youtube-remote-components`: Remote components (default: `ejs:github`)
- **Interactions**:
  - Requires `--youtube-api-key` to be specified
  - Incompatible with `--input`, `--rtsp`, `--web`, `--from-output`
  - Requires text output

---

## Language Configuration

### `--input-lang <code>`
- **Type**: Language code (e.g., `en`, `fr`, `es`, `auto`)
- **Default**: `auto` (automatic detection)
- **Special cases**:
  - For text file input (`.txt`): **Must NOT be `auto`** — must specify actual language
  - For audio sources: Can be `auto` for automatic detection
- **Examples**: `en`, `fr`, `es`, `de`, `ja`, `zh`, `pt`, etc.
- **Interactions**:
  - Text file input validation: Fails if `input-lang` is `auto` with `.txt` files
  - No interaction with transcription backend choice

### `--output-lang <code>`
- **Type**: Language code (e.g., `en`, `fr`, `es`)
- **Default**: `en` (English)
- **Controls Stage 2**:
  - If `output-lang` = `en`: Stage 2 (translation) is **skipped**
  - If `output-lang` ≠ `en`: Stage 2 (translation) **runs** to translate to target language
- **TTS language**: By default uses `output-lang` for voice synthesis
- **Can be overridden**: By `--voice-lang` for specific voice synthesis language
- **Examples**: `en`, `fr`, `es`, `de`, `ja`, `zh`, `pt`, etc.
- **Legacy alias**: `--outputlang` (deprecated but still works)
- **Interactions**:
  - Determines whether Stage 2 (translation) runs
  - Affects translation backend selection and text translation backend configuration
  - Can be overridden by `--voice-lang` for TTS only

---

## Stage 1 Outputs (English Transcription)

These options save or output the English transcription from Stage 1.

### `--scribe-text <file>`
- **Type**: File path
- **Output format**: Plain text file containing English transcription
- **When it runs**: Always runs (first stage of pipeline)
- **Typical use**: Save English transcript for review or further processing
- **Interactions**:
  - Works with all input types
  - Can be combined with `--slate-text` for bilingual output
  - Complements `--scribe-voice`

### `--scribe-voice <file>`
- **Type**: File path with audio extension (`.wav`, `.mp3`)
- **Output format**: Audio file containing English transcription read aloud
- **When it runs**: Only if voice output is requested
- **TTS engine**:
  - Default: gTTS (Google Text-to-Speech)
  - Override with `--voice-backend piper` for local Piper TTS
- **Requires**: Voice synthesis to be available (gTTS or Piper)
- **Interactions**:
  - Requires TTS engine available
  - Can be combined with `--slate-voice`
  - Uses `en_US` (English) for voice synthesis
  - Respects `--voice-backend` and `--voice-model` settings

---

## Stage 2 Outputs (Translation or Re-Output)

These options save or output the translated text (or re-output if `output-lang` = `en`).

### `--slate-text <file>`
- **Type**: File path
- **Output format**: Plain text file
- **Behavior**:
  - If `output-lang` ≠ `en`: Contains **translated** text
  - If `output-lang` = `en`: Contains **English** text (same as scribe-text, but separate file)
- **Typical use**: Save final output in target language
- **Interactions**:
  - Works with all input types
  - Can be combined with `--scribe-text` for bilingual output
  - Complements `--slate-voice`
  - Output depends on `--output-lang` setting

### `--slate-voice <file>`
- **Type**: File path with audio extension (`.wav`, `.mp3`)
- **Output format**: Audio file
- **Behavior**:
  - If `output-lang` ≠ `en`: Reads **translated** text aloud in target language
  - If `output-lang` = `en`: Reads English text aloud
- **TTS engine**:
  - Default: gTTS
  - Override with `--voice-backend piper` for local Piper TTS
- **Language for TTS**:
  - Default: `output-lang` (or `en` if output-lang is `en`)
  - Can be overridden with `--voice-lang`
- **Interactions**:
  - Requires TTS engine available
  - Language used depends on `--output-lang` unless overridden
  - Can be combined with `--scribe-voice` for bilingual voice output
  - Respects `--voice-backend` and `--voice-model` settings

---

## Scribe Options (Speech-to-Text / Stage 1 Configuration)

### `--scribe-backend <name>`
- **Type**: Choice
- **Choices**: `whispercpp`, `whispercpp-cli`, `faster-whisper`, `whisper-ctranslate2`
- **Default**: `faster-whisper`
- **Behavior**: Selects which Whisper implementation to use
- **Performance characteristics**:
  - `whispercpp`: Fast C++ implementation, requires binary
  - `whispercpp-cli`: CLI-based version of whisper.cpp
  - `faster-whisper`: Using CTranslate2 for optimization, good balance
  - `whisper-ctranslate2`: Optimized inference with CTranslate2, may need GPU setup
- **Interactions**:
  - `--scribe-model`: Backend-specific model names (e.g., `tiny`, `small`, `medium`, `large`)
  - `--magnitude-threshold`: Used by all backends for silence detection
  - Backend-specific options only apply to chosen backend (non-applicable options are ignored)

### `--scribe-model <name|path>`
- **Type**: Model name or file path
- **Default**: `medium`
- **Backend-specific values**:
  - `whispercpp`: Model name (tiny, base, small, medium, large) OR path to `.bin` file
  - `faster-whisper`: Model name (tiny, base, small, medium, large-v2, large-v3, etc.)
  - `whisper-ctranslate2`: Model name or path to local model
- **Behavior**: 
  - For `whispercpp`: Auto-downloads if `--auto-download` enabled and model not found
  - For other backends: Uses HuggingFace Hub or local path
- **Auto-download**: Only applies to `whispercpp` backend
- **Interactions**:
  - Must be compatible with chosen `--scribe-backend`
  - `--whispercpp-model-dir`: Affects where `whispercpp` looks for models
  - `--auto-download`: Enables auto-download for `whispercpp` only

### `--magnitude-threshold <float>`
- **Type**: Float (0.0 to 1.0)
- **Default**: `0.01`
- **Purpose**: Threshold for silence detection during audio processing
- **Behavior**: Frames below this amplitude are considered silence
- **Lower values**: More sensitive to quiet speech (may cause false positives)
- **Higher values**: Less sensitive (may miss quiet speech)
- **Typical range**: `0.001` to `0.05` (depends on audio quality and environment)
- **Interactions**:
  - Used by all backends
  - Can be fine-tuned with `--scribe-vad` (Silero VAD) for better results

### `--scribe-vad`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: Use Silero Voice Activity Detection for better speech detection
- **Behavior**: 
  - Detects speech vs. silence more accurately than magnitude-based detection
  - Reduces false positives and false negatives
  - Slightly increases processing time
- **Requirements**: Silero VAD library (`pip install silero-vad`)
- **Fallback**: Automatically falls back to magnitude threshold if not installed
- **Interactions**:
  - Works alongside `--magnitude-threshold` (uses `--magnitude-threshold` as fallback)
  - Works with all input types
  - Improves quality with noisy audio

### `--verbose`
- **Type**: Boolean flag
- **Action**: Enable detailed logging output
- **What gets logged**:
  - Backend configuration details
  - File operations (model downloads, directory creation)
  - Pipeline stage execution timing
  - Configuration warnings and debug info
- **Interactions**: Works with all other options

---

## Scribe Options: whisper.cpp and whisper-ctranslate2 Configuration

For whisper.cpp and whisper-ctranslate2 backend-specific options, see [WHISPER.md](WHISPER.md).

---

## Slate Options (Text Translation / Stage 2 Configuration)

Stage 2 runs text translation if `--output-lang` ≠ `en`.

### `--slate-backend <backend>`
- **Type**: Choice
- **Choices**: `googletrans`, `libretranslate`, `translategemma`, `metanllb`, `marianmt`, `none`
- **Default**: `googletrans`
- **Behavior**: Selects the text translation service for Stage 2 (language translation)
- **Options**:
  - `googletrans`: Free, no API key needed, rate-limited
  - `libretranslate`: Self-hosted or remote instance, free/paid
  - `translategemma`: Local Google TranslateGemma AI model, no API key, runs on GPU/CPU
  - `metanllb`: Local Meta NLLB model, no API key, 200+ languages, runs on GPU/CPU
  - `marianmt`: Local Helsinki-NLP Marian MT models, no API key, lightweight, runs on GPU/CPU
  - `none`: Skips text translation (Stage 2)
- **Interactions**:
  - Only used if `--output-lang` ≠ `en` (Stage 2 runs)
  - `--libretranslate-url`: Required if backend is `libretranslate`
  - `--slate-model`: Selects the model variant for `translategemma`, `metanllb`, or `marianmt`
  - Parallel to transcription backend choice (independent)

### `--libretranslate-url <url>`
- **Type**: URL
- **Example**: `http://localhost:5000` or `https://libretranslate.example.com`
- **Required**: When `--slate-backend libretranslate`
- **Purpose**: URL of LibreTranslate instance (self-hosted or remote)
- **Behavior**: Sends text translation requests to this endpoint
- **Interactions**:
  - Only used if `--slate-backend libretranslate`
  - Must be running and accessible when translation is needed

### `--slate-model <model>`
- **Type**: HuggingFace model name
- **Default**: Depends on `--slate-backend`:
  - `translategemma`: `google/translategemma-12b-it`
  - `metanllb`: `facebook/nllb-200-1.3B`
  - `marianmt`: Auto-derived from language pair (e.g., `Helsinki-NLP/opus-mt-en-fr`)
- **Available models (translategemma)**:
  - `google/translategemma-4b-it` — fastest, smallest (~8GB)
  - `google/translategemma-12b-it` — balanced (default, ~22GB)
  - `google/translategemma-27b-it` — highest quality, largest (~54GB)
- **Available models (metanllb)**:
  - `facebook/nllb-200-distilled-600M` — fastest, smallest
  - `facebook/nllb-200-1.3B` — balanced (default)
  - `facebook/nllb-200-3.3B` — higher quality, larger
- **Available models (marianmt)**: Any `Helsinki-NLP/opus-mt-{source}-{target}` model
- **Purpose**: Selects the model variant for the chosen translation backend
- **Behavior**: Model is downloaded automatically on first use and cached
- **Interactions**:
  - Only used with `--slate-backend translategemma`, `metanllb`, or `marianmt`
  - GPU strongly recommended for `translategemma` and `metanllb`
  - See [TRANSLATEGEMMA_SETUP.md](TRANSLATEGEMMA_SETUP.md) for `translategemma` setup

---

## Voice Options (Text-to-Speech / Stage 3 Configuration)

Stage 3 generates voice output if `--scribe-voice` or `--slate-voice` is specified.

### `--voice-backend <backend>`
- **Type**: Choice
- **Choices**: `gtts`, `piper`, `cosyvoice`, `custom`, `auto`
- **Default**: `auto`
- **Purpose**: Select the TTS engine for voice synthesis
- **Options**:
  - `gtts`: Google Text-to-Speech (network call, no installation needed)
  - `piper`: Local Piper TTS (fast, private, works offline)
  - `cosyvoice`: CosyVoice TTS (advanced multilingual TTS with zero-shot voice cloning)
  - `custom`: Qwen3-TTS (CustomVoice or Base models for voice cloning)
  - `auto`: Automatically selects best available backend (prefers Piper, falls back to gTTS)
- **Benefits of `piper`**:
  - Faster (no network calls)
  - Private (no data sent to external services)
  - Works offline
  - Better voice quality for some languages
- **Benefits of `cosyvoice`**:
  - State-of-the-art multilingual TTS (9 languages, 18+ Chinese dialects)
  - Zero-shot voice cloning with `--voice-match`
  - High-quality, natural speech synthesis
  - Advanced pronunciation control
- **Benefits of `custom`**:
  - Qwen3-TTS CustomVoice and Base models
  - Voice cloning using Base model with `--voice-match`
  - Multilingual support (English, Chinese, Japanese, Korean, and more)
- **Requirements**: 
  - For `piper`: Piper must be installed (`pip install piper-tts` or binary)
  - For `cosyvoice`: CosyVoice must be installed (`pip install -e .[cosyvoice]`)
  - For `custom`: qwen-tts must be installed (`pip install -e .[custom]`)
- **Fallback**: Automatically falls back to gTTS if selected backend not found
- **Interactions**:
  - Works with `--scribe-voice` and `--slate-voice`
  - `--voice-model`: Selects which voice model to use when `--voice-backend piper`, `cosyvoice`, or `custom`
  - `--voice-match`: Enables voice cloning for `cosyvoice`/`custom` or voice matching for `piper`
  - `--voice-lang`: May override voice language

### `--voice-model <voice-model>`
- **Type**: Voice model identifier
- **Default**: `en_US-lessac-medium`
- **Format**: 
  - For Piper: `{language_code}-{speaker_name}-{quality}` (e.g., `en_US-lessac-medium`)
  - For CosyVoice: Model name or path (e.g., `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`)
  - For custom: Qwen3-TTS model name (e.g., `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)
- **Examples**:
  - **Piper**:
    - `en_US-lessac-medium`: English (US) - clear male voice
    - `en_US-norman-medium`: English (US) - different speaker
    - `en_GB-amy-medium`: English (British)
    - `es_ES-carla-x-low`: Spanish
    - `fr_FR-siwis-medium`: French
  - **CosyVoice**:
    - `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`: Latest CosyVoice 3.0 model
    - `FunAudioLLM/CosyVoice2-0.5B`: CosyVoice 2.0 model
  - **Custom (Qwen3-TTS)**:
    - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`: CustomVoice model (default speaker)
    - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`: Base model (for voice cloning with `--voice-match`)
- **Finding voices**: 
  - Piper: Check Piper documentation or available voice list
  - CosyVoice: Check HuggingFace model hub
  - Custom: Check HuggingFace model hub for Qwen3-TTS models
- **Interactions**:
  - Used when `--voice-backend piper`, `--voice-backend cosyvoice`, or `--voice-backend custom` is specified
  - `--voice-lang`: Does not override voice, but should match voice language

### `--voice-lang <code>`
- **Type**: Language code (e.g., `en`, `fr`, `es`)
- **Default**: Determined by `--output-lang` (or `en` if output-lang is `en`)
- **Purpose**: Override the language used for voice synthesis
- **Use cases**:
  - Output text is in language A, but synthesize voice in language B
  - Force English pronunciation despite non-English output text
- **Interactions**:
  - Only used if voice output is requested
  - Works with both gTTS and Piper TTS
  - Should match or be compatible with `--voice-model` if using Piper
  - Typically doesn't override text language (just voice synthesis)

### `--voice-match`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: 
  - For **Piper**: Automatically select the closest matching Piper TTS voice based on input speaker characteristics
  - For **CosyVoice**: Enable zero-shot voice cloning using reference audio from input
  - For **Custom**: Enable voice cloning using Qwen3-TTS Base model with reference audio
- **How it works**:
  - **Piper**:
    1. Extracts pitch and spectral features from input audio
    2. Estimates gender (Male: <140Hz, Female: >180Hz)
    3. Selects the closest matching Piper voice for the target language
    4. Falls back to default voice if no suitable match found
  - **CosyVoice**:
    1. Uses input audio segments as reference for voice cloning
    2. Synthesizes output speech in the same voice as the input speaker
    3. Works across different languages (cross-lingual cloning)
  - **Custom**:
    1. Uses Qwen3-TTS Base model with reference audio for voice cloning
    2. Synthesizes output speech mimicking the input speaker's voice
- **Requirements**: 
  - For Piper: `pip install -e .[piper]`; Piper TTS must be installed
  - For CosyVoice: `pip install -e .[cosyvoice]`; CosyVoice must be installed
  - For Custom: `pip install -e .[custom]`; qwen-tts must be installed
- **Interactions**:
  - Works with `--slate-voice` and Piper, CosyVoice, or Custom TTS
  - Pairs well with `--voice-backend piper`, `--voice-backend cosyvoice`, or `--voice-backend custom`
  - See [VOICE_MATCHING.md](VOICE_MATCHING.md) for Piper voice matching details

---

## Audio Processing

### `--window-seconds <float>`
- **Type**: Float > 0
- **Default**: `5.0`
- **Unit**: Seconds
- **Purpose**: Size of audio chunks for processing
- **Behavior**: Pipeline processes audio in this-sized windows
- **Considerations**:
  - Larger windows = lower granularity, fewer API calls, more initial latency
  - Smaller windows = higher granularity, more API calls, lower latency
  - Typical range: 3.0–10.0 seconds
- **Validation**: Must be > 0; returns error otherwise
- **Interactions**:
  - `--overlap-seconds`: Must be less than this value
  - Affects real-time latency of output

### `--overlap-seconds <float>`
- **Type**: Float ≥ 0
- **Default**: `0.0`
- **Unit**: Seconds
- **Purpose**: Overlap between consecutive audio chunks
- **Behavior**: Creates overlap for context continuity
- **Use cases**:
  - `0.0`: No overlap (default, no repeated processing)
  - `1.0-2.0`: Overlap for better context at chunk boundaries
- **Performance impact**: Creates duplicate processing but may improve quality
- **Validation**: Must be ≥ 0 and < `--window-seconds`
- **Interactions**:
  - Must be less than `--window-seconds`
  - Affects API call count and processing overhead

---

## MQTT Publishing

For MQTT publishing options, see [MQTT.md](MQTT.md).

---

## Web Server Configuration

These options apply when `--web` is specified.

### `--web-host <address>`
- **Type**: IP address or hostname
- **Default**: `0.0.0.0` (listen on all interfaces)
- **Examples**:
  - `0.0.0.0`: All IPv4 interfaces (external + localhost)
  - `127.0.0.1`: Localhost only (not accessible from network)
  - `::1`: IPv6 localhost
  - `192.168.1.100`: Specific network interface
- **Purpose**: Which network interface the web server listens on
- **Security**:
  - `127.0.0.1`: Most secure (local only)
  - `0.0.0.0`: Accessible from network (use with HTTPS)
- **Interactions**:
  - `--web-ssl-self-signed`: Uses this as `common_name` for certificate if generating self-signed cert

### `--web-port <int>`
- **Type**: Integer (1-65535)
- **Default**: `8443` (HTTPS)
- **Common choices**:
  - `8443`: Default HTTPS
  - `8444`: Alternative HTTPS
  - `80`: HTTP (not recommended, requires admin on Linux)
  - `443`: Standard HTTPS (requires admin on Linux)
- **Examples**: `--web-port 8444`, `--web-port 9000`
- **Interactions**:
  - Works with all web server options

### `--web-ssl-cert <path>`
- **Type**: File path
- **Format**: PEM format SSL certificate
- **Requirement**: Must be provided with `--web-ssl-key` if using custom certs
- **Alternative**: Use `--web-ssl-self-signed` to auto-generate
- **Purpose**: HTTPS certificate file
- **Behavior**:
  - Required for HTTPS server
  - Must match private key
  - Can be self-signed or CA-signed
- **Interactions**:
  - Must be paired with `--web-ssl-key`
  - Cannot mix: Either provide both OR use `--web-ssl-self-signed`, not both
  - Error if `--web-ssl-cert` without `--web-ssl-key`

### `--web-ssl-key <path>`
- **Type**: File path
- **Format**: PEM format private key
- **Requirement**: Must be provided with `--web-ssl-cert`
- **Alternative**: Use `--web-ssl-self-signed` to auto-generate
- **Purpose**: HTTPS private key file
- **Behavior**:
  - Required for HTTPS server
  - Must match certificate
  - Must be kept private
- **Security**: Never commit to version control
- **Interactions**:
  - Must be paired with `--web-ssl-cert`
  - Cannot mix: Either provide both OR use `--web-ssl-self-signed`, not both
  - Error if `--web-ssl-key` without `--web-ssl-cert`

### `--web-ssl-self-signed`
- **Type**: Boolean flag
- **Purpose**: Auto-generate self-signed SSL certificate
- **Behavior**:
  - Generates `selfsigned.crt` and `selfsigned.key` in current directory
  - Only generates if files don't already exist
  - Uses `--web-host` as certificate common name
- **When to use**:
  - Development and testing
  - Internal deployment without CA-signed certs
  - Quick setup without manual cert generation
- **Browser warning**: Browsers will warn about self-signed certs (expected)
- **Interactions**:
  - Cannot be used with `--web-ssl-cert` and `--web-ssl-key` simultaneously
  - Error if both cert files exist and cert/key options are provided
  - `--web-host`: Used as common name in certificate

---

## YouTube Configuration

These options apply when `--youtube-url` is specified.

### `--youtube-api-key <key>`
- **Type**: YouTube Data API v3 key
- **Required**: When using `--youtube-url`
- **How to get**: 
  1. Go to Google Cloud Console
  2. Create/select a project
  3. Enable YouTube Data API v3
  4. Create API credentials (API key)
- **Purpose**: Authentication for YouTube API
- **Limitations**: 
  - Subject to daily quota limits
  - Free tier has rate limits
- **Security**: Consider using environment variable
- **Interactions**:
  - Required with `--youtube-url`
  - Error if `--youtube-url` without API key

### `--youtube-js-runtime <path>`
- **Type**: File path or command
- **Default**: `node`
- **Examples**:
  - `node`: Uses system Node.js
  - `C:\nodejs\node.exe`: Explicit path on Windows
  - `node:C:\path\to\node.exe`: Windows syntax variant
- **Purpose**: JS runtime for yt-dlp video downloading
- **When needed**: If system Node.js is not in PATH
- **Interactions**:
  - Optional; only needed if default `node` not available

### `--youtube-remote-components <spec>`
- **Type**: String specification
- **Default**: `ejs:github`
- **Purpose**: yt-dlp remote components for plugin system
- **Examples**: `ejs:github`, `ejs:gitlab`, custom URLs
- **Advanced**: Usually not needed to change
- **Interactions**:
  - Optional; default works for most cases

---

## System Audio Capture (Windows Only)

These options apply when `--from-output` is specified (Windows WASAPI loopback).

### `--output-device <name>`
- **Type**: Device name (string)
- **Platform**: Windows only
- **Purpose**: Select specific WASAPI loopback device
- **Behavior**: Focuses capture on specified device
- **Finding device names**:
  - Use `--list-output-devices` to see available options
  - Look for "Stereo Mix" or similar loopback device
- **Example**: `--output-device "Stereo Mix"`
- **Interactions**:
  - Only used with `--from-output`
  - Must match exactly (case-sensitive)
  - If not specified, uses default loopback device

### `--list-output-devices`
- **Type**: Boolean flag
- **Platform**: Windows only
- **Behavior**: Lists available WASAPI loopback devices and exits
- **Output format**: Prints device names to console
- **Use cases**:
  - Discover available loopback devices before running capture
  - Verify device names for `--output-device` option
- **Example**:
  ```bash
  anytran --list-output-devices
  ```
- **Interactions**: Mutually exclusive with normal pipeline execution

---

## Logging & Output

### `--chat-log <directory>`
- **Type**: Directory path
- **Default**: `./chat`
- **Purpose**: Directory for chat/transcription logs
- **Format**: Rotating hourly log files with timestamp, IP, and transcription
- **Behavior**:
  - Auto-creates directory if doesn't exist
  - One file per hour (rotating)
  - Logs include timestamp, source IP, transcription text
- **Typical use**: Archive transcriptions for audit/review
- **Interactions**:
  - Auto-created on startup if doesn't exist
  - Used by all streaming modes
  - `--verbose` shows directory creation

### `--timers`
- **Type**: Boolean flag
- **Purpose**: Print timing summary by stage only
- **Output includes**:
  - Time for Stage 1 (transcription)
  - Time for Stage 2 (translation)
  - Time for Stage 3 (TTS)
  - Total time per chunk
- **Typical output**: Printed to stderr or logs
- **Use cases**: Performance profiling, debugging slow operations
- **Interactions**: Works with all other options

### `--timers-all`
- **Type**: Boolean flag
- **Purpose**: Print all timing summaries (superset of `--timers`)
- **Output includes**:
  - Full timing summary (all measured keys)
  - Timing summary by stage (same as `--timers`)
  - Translate overhead (difference between total and backend time)
- **Typical output**: Printed to stderr or logs
- **Use cases**: Deep performance analysis, backend overhead investigation
- **Interactions**: Works with all other options

---

## Utilities & Diagnostics

### `--verbose`
- **Type**: Boolean flag
- **Purpose**: Enable detailed logging
- **Output includes**:
  - Backend configuration details
  - Environment variable values
  - File operations (model downloads, directory creation)
  - Pipeline execution details
  - Warning and debug messages
- **Performance impact**: Minimal (just more output)
- **Interactions**: Works with all other options

### `--keep-temp`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: Preserve temporary files after processing (for debugging)
- **Behavior**: By default, temporary audio and intermediate files are cleaned up automatically
- **Interactions**: Works with all other options

### `--dedup`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: Enable deduplication of text output
- **Behavior**: Filters out repeated or near-duplicate lines from the output text files
- **Use cases**: Reduce noise when processing repetitive audio or when windows overlap
- **Interactions**: Applies to both `--scribe-text` and `--slate-text` output

### `--lang-prefix`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: Prefix each output text line with the language name
- **Output format**: `English: <text>` or `French: <text>` depending on language
- **Use cases**: Bilingual log files, side-by-side transcription/translation
- **Interactions**: Works with all output types and input sources

### `--no-norm`
- **Type**: Boolean flag
- **Default**: Disabled (normalization is ON by default)
- **Purpose**: Disable text normalization before writing output to file
- **Behavior**: By default, output text is normalized (whitespace cleanup, punctuation). This flag disables that step.
- **Use cases**: Preserve raw transcription/translation output exactly as received
- **Interactions**: Applies to all text file outputs

### `--batch-input-text <N>`
- **Type**: Integer ≥ 0
- **Default**: `0` (no batching)
- **Purpose**: Batch N lines or sentences together when processing text file input
- **Behavior**: Groups N lines into a single translation request instead of translating line by line
- **Benefits**: Reduces API call overhead; may improve context-aware translations
- **Interactions**:
  - Only applies to `--input` with text file (`.txt`)
  - `0` means no batching (each line translated individually)

---

## Text File Translation Loop Options

These options apply when `--input` is a text file and enable iterative back-translation.

### `--looptran <N>`
- **Type**: Integer ≥ 0
- **Default**: `0` (disabled)
- **Purpose**: Repeat translation N additional times, alternating language direction each pass
- **Requirements**:
  - `--input` must be a text file (`.txt`)
  - `--slate-text` must be specified
  - `--input-lang` and `--output-lang` must differ
- **Behavior**:
  1. Initial pass: translates input from `--input-lang` → `--output-lang`, writes to `--slate-text`
  2. Pass 1: translates `--slate-text` (output-lang → input-lang), writes to `<slate>_1.<ext>`
  3. Pass 2: translates `<slate>_1` back, writes to `<slate>_2.<ext>`
  4. … continues N times, alternating languages
- **Output files**: `slate.txt`, `slate_1.txt`, `slate_2.txt`, …
- **Use cases**: Measure translation stability, quality testing, back-translation verification
- **Interactions**:
  - Requires `--input` to be a `.txt` file
  - Ignored for audio input
  - Use with `--tran-converge` to stop early when stable
  - See [LOOPTRAN.md](LOOPTRAN.md) for detailed documentation and examples

### `--tran-converge`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: Enable early stopping in `--looptran` when translation has converged
- **Behavior**: After each iteration i ≥ 2, compares the output with the output from two iterations back (same language direction). If they are identical, reports convergence and stops looping early.
- **Example**: With `--looptran 10 --tran-converge`, if `slate_4.txt` exactly matches `slate_2.txt`, the loop stops at iteration 4 instead of running all 10 passes.
- **Use cases**: Automatically detect when back-translation stabilizes without specifying an exact iteration count
- **Interactions**:
  - Only meaningful when `--looptran` > 0
  - Requires `--input` to be a `.txt` file (same as `--looptran`)
  - See [LOOPTRAN.md](LOOPTRAN.md) for detailed documentation and examples

---

## Configuration File Options

These options allow you to load settings from a config file or generate a template config file. They are processed before all other options.

### `--config <path>`
- **Type**: File path
- **Format**: JSON (`.json`) or TOML (`.toml`) — determined by file extension; any other extension is treated as JSON
- **Purpose**: Load default settings from a config file before applying CLI options
- **Behavior**:
  - CLI arguments always override values from the config file
  - Config file provides defaults; any option passed on the command line takes precedence
  - When a config file is loaded, non-default settings are printed before the pipeline starts
- **JSON example** (`anytran.json`):
  ```json
  {
    "output_lang": "fr",
    "slate_backend": "libretranslate",
    "libretranslate_url": "http://localhost:5000"
  }
  ```
- **TOML example** (`anytran.toml`):
  ```toml
  output_lang = "fr"
  slate_backend = "libretranslate"
  libretranslate_url = "http://localhost:5000"
  ```
- **Key names**: Use underscore-separated names matching the argparse `dest` (e.g., `output_lang`, not `output-lang`)
- **Interactions**:
  - Can be combined with `--genconfig` to create and then load a config file
  - All standard CLI options can be specified in the config file

### `--genconfig [path]`
- **Type**: Optional file path
- **Default path**: `anytran.json` (if no path given)
- **Special value**: `-` prints JSON to stdout instead of writing to a file
- **Format**: JSON (`.json`) or TOML (`.toml`) — determined by file extension; default is JSON
- **Purpose**: Generate a config file containing all current settings (including defaults) and exit
- **Behavior**:
  - Writes all parsed argument values to the specified file
  - Exits immediately after writing — no pipeline is run
  - If `--config` is also specified, the resulting file reflects config-file values merged with any CLI overrides
  - If the file already exists, it is overwritten without warning
- **Examples**:
  ```bash
  # Generate a default JSON config file named anytran.json
  anytran --genconfig

  # Generate a TOML config file
  anytran --genconfig anytran.toml

  # Preview config as JSON on stdout
  anytran --genconfig -

  # Generate a config with specific settings pre-applied
  anytran --output-lang fr --slate-backend libretranslate --genconfig my_config.json
  ```
- **Interactions**:
  - Can be combined with `--config` to merge an existing config with CLI overrides and save the result
  - Does not require an input source (exits before pipeline validation)

---

## Voice Table Generation

These options generate or update the voice table JSON file used by the voice matching feature. They are processed early and cause the program to exit after generation without requiring an input source.

### `--voice-table-gen`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: Generate or update the voice table JSON file with voice features and exit
- **Behavior**:
  - Scans available Piper TTS voices for the specified languages
  - Extracts voice features (pitch, gender, spectral characteristics) for each voice
  - Writes results to the output JSON file
  - Exits immediately after generation — no pipeline is run
- **Requirements**: `pip install -e .[piper]`; Piper TTS and librosa must be installed
- **Examples**:
  ```bash
  # Generate voice table for French (default)
  anytran --voice-table-gen

  # Generate for French and English
  anytran --voice-table-gen --voice-table-lang fr,en

  # Generate for all supported languages
  anytran --voice-table-gen --voice-table-lang all

  # Generate to a custom output path
  anytran --voice-table-gen --voice-table-lang fr,en --voice-table-output ./my_voice_table.json
  ```
- **Interactions**:
  - Does not require an input source (exits before pipeline validation)
  - Use `--voice-table-lang` to select languages
  - Use `--voice-table-output` to set the output file path

### `--voice-table-lang <languages>`
- **Type**: Comma-separated string of language codes
- **Default**: `fr`
- **Special value**: `all` — process all supported languages
- **Purpose**: Specify which languages to include when generating the voice table
- **Examples**:
  - `fr` — French only
  - `fr,en` — French and English
  - `fr,en,de,es` — French, English, German, Spanish
  - `all` — all supported languages
- **Interactions**:
  - Only used when `--voice-table-gen` is specified

### `--voice-table-output <path>`
- **Type**: File path
- **Default**: `src/anytran/voice_table.json`
- **Purpose**: Set the output path for the generated voice table JSON file
- **Behavior**:
  - Creates or overwrites the file at the specified path
  - The generated file is used by `--voice-match` at runtime to select voices
- **Examples**:
  - `--voice-table-output ./my_voice_table.json`
  - `--voice-table-output /custom/path/voice_table.json`
- **Interactions**:
  - Only used when `--voice-table-gen` is specified

---

## Inverted Boolean Options

Many boolean flags have a paired `--no-*` (or `--*`) counterpart that explicitly disables the feature. These are useful when a config file sets a flag to `true` and you need to override it from the CLI, or when scripts need to be explicit about the state of a flag.

### General Pattern
- `--flag` enables the feature (sets value to `True`)
- `--no-flag` disables the feature (sets value to `False`)

CLI arguments always take precedence over config file values, so `--no-flag` can be used to turn off a feature that was enabled in a config file.

### Complete List of Inverted Boolean Pairs

| Enable | Disable | Default |
|--------|---------|---------|
| `--scribe-vad` | `--no-scribe-vad` | Disabled |
| `--auto-download` | `--no-auto-download` | Enabled (auto-download on) |
| `--whispercpp-cli-detect-lang` | `--no-whispercpp-cli-detect-lang` | Disabled |
| `--voice-match` | `--no-voice-match` | Disabled |
| `--web-ssl-self-signed` | `--no-web-ssl-self-signed` | Disabled |
| `--verbose` | `--no-verbose` | Disabled |
| `--timers` | `--no-timers` | Disabled |
| `--timers-all` | `--no-timers-all` | Disabled |
| `--lang-prefix` | `--no-lang-prefix` | Disabled |
| `--keep-temp` | `--no-keep-temp` | Disabled |
| `--dedup` | `--no-dedup` | Disabled |
| `--norm` | `--no-norm` | Enabled (normalization on) |
| `--input-norm` | `--no-input-norm` | Enabled (input normalization on) |

### `--no-input-norm`
- **Type**: Boolean flag
- **Default**: Disabled (input normalization is **on** by default)
- **Purpose**: Disable normalization of the original input text file before processing
- **Behavior**: When using `--input` with a `.txt` file, the input text is normalized by default (whitespace cleanup, punctuation). `--no-input-norm` disables that step for the input file only.
- **Note**: This is separate from `--no-norm`, which controls **output** text normalization
- **Interactions**:
  - Only applies to `--input` with `.txt` files
  - Use `--input-norm` to re-enable input normalization explicitly (e.g., to override a config file that sets `no_input_norm: true`)

---

## Common Option Interactions & Constraints

### Input Source Selection
- Exactly **one** input source must be specified: `--input`, `--rtsp`, `--web`, `--from-output`, or `--youtube-url`
- Specifying multiple input sources causes an error

### Output Requirements
- **`--from-output`**: Requires at least one text output (`--scribe-text` or `--slate-text`)
- **`--youtube-url`**: Requires at least one text output (`--scribe-text` or `--slate-text`)
- **`--rtsp`**: Requires at least one output (text, voice, or MQTT)
- **`--web`**: Works with any combination of outputs

### Language Workflow
1. Input → transcribed to English (Stage 1)
2. If `--output-lang` ≠ `en` → translate to target language (Stage 2)
3. Text → voice in appropriate language (Stage 3, if voice output requested)

### Backend Coherence
- Transcription backend (`--scribe-backend`) and text translation backend (`--slate-backend`) are **independent**
- Choose transcription backend for audio → text quality
- Choose text translation backend based on language pair and API availability

### SSL Certificate For Web Mode
- **Provide both** `--web-ssl-cert` and `--web-ssl-key`, OR
- **Use** `--web-ssl-self-signed` to auto-generate, OR
- Provide **neither** (not recommended; HTTPS required by modern browsers for mic access)
- **Cannot mix**: Don't provide cert files AND `--web-ssl-self-signed` at the same time

### MQTT Topic Count Matching
- If using multiple `--rtsp` streams AND `--mqtt-topic-names`:
  - Number of topic names **must equal** number of RTSP streams
  - Otherwise, error is raised

### Text Input File Special Case
- If input is `.txt` file:
  - `--input-lang` **cannot be `auto`** (must specify language)
  - Stage 1 (transcription) is skipped
  - Starts at Stage 2 (translation) if `--output-lang` ≠ input language

### Piper TTS Configuration
- If `--voice-backend piper` is set but Piper not installed:
  - Automatic fallback to gTTS
  - Warning printed to console
  - No error/failure

### Silero VAD Configuration
- If `--scribe-vad` is set but Silero VAD not installed:
  - Automatic fallback to magnitude threshold
  - Warning printed to console
  - No error/failure

---

## Environment Variables

Some options can be set via environment variables (useful for containerization and CI/CD):

| Environment Variable | Corresponding Option | Notes |
|---|---|---|
| `WHISPERCPP_BIN` | `--whispercpp-bin` | Path to whisper.cpp binary |
| `WHISPERCPP_MODEL_DIR` | `--whispercpp-model-dir` | Directory for whisper.cpp models |
| `WHISPERCPP_THREADS` | `--whispercpp-threads` | Number of threads (rarely needed) |
| `WHISPER_CTRANSLATE2_DEVICE` | `--whisper-ctranslate2-device` | Device: `auto`, `cuda`, `cpu` |
| `WHISPER_CTRANSLATE2_DEVICE_INDEX` | `--whisper-ctranslate2-device-index` | GPU index |
| `WHISPER_CTRANSLATE2_COMPUTE_TYPE` | `--whisper-ctranslate2-compute-type` | Precision: `default`, `float16`, `int8` |

**Note**: Command-line options always take precedence over environment variables.

---

## Quick Reference: Common Workflows

### Transcribe File to Text
```bash
anytran --input audio.mp3 --scribe-text transcript.txt
```

### Translate YouTube to French with Voice
```bash
anytran --youtube-url "https://youtube.com/watch?v=..." \
  --youtube-api-key YOUR_KEY \
  --output-lang fr \
  --slate-text french.txt \
  --slate-voice french.wav \
  --voice-backend piper
```

### Real-time RTSP Streaming with MQTT
```bash
anytran --rtsp rtsp://camera.example.com/stream \
  --scribe-text output.txt \
  --mqtt-broker mqtt.example.com \
  --mqtt-port 1883 \
  --mqtt-topic audio/transcription
```

### Web Server for Microphone Input
```bash
anytran --web \
  --web-port 8443 \
  --web-ssl-self-signed \
  --output-lang es \
  --slate-text spanish.txt \
  --voice-backend piper \
  --voice-model es_ES-carla-medium
```

### Translate Text Input
```bash
anytran --input my_text.txt \
  --input-lang es \
  --output-lang en \
  --slate-text english.txt
```

### System Audio Capture (Windows)
```bash
anytran --from-output \
  --output-device "Stereo Mix" \
  --scribe-text transcript.txt \
  --slate-text translated.txt \
  --output-lang fr
```

### GPU-Accelerated Transcription
```bash
anytran --input audio.mp4 \
  --scribe-backend whisper-ctranslate2 \
  --whisper-ctranslate2-device cuda \
  --whisper-ctranslate2-compute-type float16 \
  --scribe-text transcript.txt
```

---

## Error Messages & Troubleshooting

### "Error: Input file not found"
- **Cause**: `--input` file doesn't exist
- **Fix**: Verify file path and that file is readable

### "Error: --input-lang must be specified (not 'auto') when using text file input"
- **Cause**: Using `.txt` file with `--input-lang auto`
- **Fix**: Specify actual language: `--input-lang fr`, `--input-lang es`, etc.

### "Error: Number of --mqtt-topic-names must match number of --rtsp streams"
- **Cause**: Mismatch between RTSP stream count and topic count
- **Fix**: Add or remove topic names to match stream count

### "Error: --web-ssl-cert and --web-ssl-key must be provided together"
- **Cause**: Only one of cert or key provided
- **Fix**: Provide both OR use `--web-ssl-self-signed`

### "Warning: --voice-backend piper specified but Piper not found"
- **Cause**: Piper TTS not installed
- **Fix**: Install with `pip install piper-tts` OR use `--voice-backend gtts`

### "Warning: --scribe-vad specified but Silero VAD not installed"
- **Cause**: Silero VAD library not installed
- **Fix**: Install with `pip install silero-vad` OR remove `--scribe-vad` flag

### "Error: whispercpp model file not found"
- **Cause**: whisper.cpp model not found and auto-download disabled
- **Fix**: Use `--auto-download` OR manually download and specify path

### "--youtube-url requires --youtube-api-key"
- **Cause**: Missing YouTube API key
- **Fix**: Provide `--youtube-api-key YOUR_KEY`

---

**Last Updated**: February 2026
