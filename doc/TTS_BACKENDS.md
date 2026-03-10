# TTS (Text-to-Speech) Backends

This document describes all available TTS backends for voice synthesis in anytran.

Voice output is produced in **Stage 3** when `--scribe-voice` or `--slate-voice` is specified.  
The backend is selected with `--voice-backend`.

---

## Backend Summary

| Backend    | Type              | Offline | Voice Cloning | Install                          |
|------------|-------------------|---------|---------------|----------------------------------|
| `auto`     | auto-select       | ✓ (if Piper available) | –    | –                                |
| `gtts`     | Cloud             | ✗       | ✗             | included (base install)          |
| `piper`    | Local neural      | ✓       | ✗             | `pip install -e .[piper]`        |
| `custom`   | Local neural (Qwen3-TTS) | ✓ | ✓ (Base model) | `pip install -e .[custom]` |
| `fish`     | Local neural (fish-speech) | ✓ | ✓          | `pip install -e .[fish]`         |
| `indextts` | Local neural (IndexTTS)    | ✓ | ✓          | see [IndexTTS install](#indextts-indexteamindextts-2) |
| `coqui`    | Local neural (coqui-tts XTTS v2) | ✓ | ✓     | `pip install -e .[coqui]`        |

---

## `auto` (default)

**Default behavior**: prefers Piper TTS if installed, otherwise falls back to gTTS automatically.

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav
# --voice-backend auto is the default; no flag required
```

---

## `gtts` — Google Text-to-Speech

Cloud-based TTS provided by Google. No installation is required beyond the base anytran install.

- **Requires**: Internet connection
- **Languages**: Wide language support
- **Voice cloning**: Not supported

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav --voice-backend gtts
```

---

## `piper` — Local Piper TTS

Fast, offline neural TTS with a large selection of voices and languages.

- **Install**: `pip install -e .[piper]`
- **Offline**: Yes
- **Voice cloning**: Not supported (fixed voices only)
- **Fallback**: Automatically falls back to gTTS if Piper is not installed

### Selecting a voice

Use `--voice-model` with a Piper voice name in `{language_code}-{speaker_name}-{quality}` format:

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend piper --voice-model fr_FR-siwis-medium
```

Common voices:
- `en_US-lessac-medium` — English (US), clear male voice (default)
- `en_US-norman-medium` — English (US), alternative speaker
- `en_GB-amy-medium` — English (British)
- `es_ES-carla-x-low` — Spanish
- `fr_FR-siwis-medium` — French

See the [Piper voice list](https://github.com/rhasspy/piper/blob/master/VOICES.md) for all available voices.

### Auto voice matching

Use `--voice-match` to automatically select the best Piper voice for the input speaker:

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend piper --voice-match
```

See [VOICE_MATCHING.md](VOICE_MATCHING.md) for details.

---

## `custom` — Qwen3-TTS

Local neural TTS based on Alibaba's [Qwen3-TTS](https://huggingface.co/Qwen) models.  
Supports both a built-in speaker (CustomVoice) and zero-shot voice cloning (Base).

- **Install**: `pip install -e .[custom]`
- **Offline**: Yes (after initial model download from HuggingFace)
- **GPU recommended**: Yes (`device_map="auto"` will use GPU if available)

### Models

| Model | Description |
|-------|-------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Default. Uses built-in "Ryan" speaker. |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Enables voice cloning with reference audio. |

Specify the model with `--voice-model`:

```bash
# Default CustomVoice model (built-in speaker)
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend custom

# Base model with voice cloning via --voice-match
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend custom \
  --voice-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --voice-match
```

### Language support

Qwen3-TTS supports many languages including English (`en`), Chinese (`zh-CN`, `zh-TW`), French (`fr`), Spanish (`es`), German (`de`), Japanese (`ja`), Korean (`ko`), and more. The `--output-lang` value is automatically mapped to the correct language name.

---

## `fish` — fish-speech

Local neural TTS based on [fish-speech](https://github.com/fishaudio/fish-speech) by fishaudio.  
Supports zero-shot voice cloning with a reference audio clip and its transcript.

- **Install**: `pip install -e .[fish]`
- **Offline**: Yes (after initial model download from HuggingFace)
- **GPU recommended**: Yes

### Models

| Model | Description |
|-------|-------------|
| `fishaudio/s1-mini` or `fishaudio/openaudio-s1-mini` | Default. Compact, fast. |
| `fishaudio/fish-speech-1.5` | Alternative model. |

```bash
# Default model
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend fish

# Specific model
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend fish --voice-model fishaudio/fish-speech-1.5
```

### Voice cloning

When `--voice-match` is enabled, anytran automatically supplies a reference audio clip from the input stream to fish-speech for voice cloning:

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend fish --voice-match
```

The reference audio is resampled to 44.1 kHz (as required by fish-speech) before being passed to the model.

---

## `indextts` — IndexTTS

Local neural TTS based on [IndexTTS](https://github.com/index-tts/index-tts) (IndexTeam/IndexTTS-2).  
Designed for high-quality voice cloning from a short speaker prompt.

### Installation

IndexTTS is not published on PyPI. Install it manually first, then install the anytran extra:

```bash
GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/index-tts/index-tts.git
pip install "anytran[index-tts]"
```

> The `GIT_LFS_SKIP_SMUDGE=1` flag skips large example audio files tracked by Git LFS — these are not needed at runtime.

### Models

| Model | Description |
|-------|-------------|
| `IndexTeam/IndexTTS-2` | Default. Downloaded automatically via HuggingFace Hub. |

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend indextts
```

### Voice cloning

IndexTTS clones the speaker's voice from a reference audio prompt.  
Use `--voice-match` to automatically supply a speaker prompt from the input stream:

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend indextts --voice-match
```

Without `--voice-match`, anytran will attempt synthesis without a reference prompt. Most IndexTTS deployments require a prompt, so synthesis may fail in that case.

---

## `coqui` — coqui-tts (XTTS v2)

Local neural TTS based on [coqui-tts](https://pypi.org/project/coqui-tts/), the Python 3.12-compatible
maintained fork of the original coqui-ai/TTS library.  
Uses the XTTS v2 model for high-quality, multilingual speech synthesis with zero-shot voice cloning.

- **Install**: `pip install "anytran[coqui]"`
- **Offline**: Yes (after initial model download)
- **Python 3.12**: ✓ fully supported (uses `coqui-tts`, not the unmaintained `TTS` package)
- **GPU recommended**: Yes (CUDA auto-detected; falls back to CPU)

> **Note on the original coqui-ai/TTS**: The original package (`pip install TTS`) does not support
> Python 3.12 — it raises `RuntimeError: TTS requires python >= 3.9 and < 3.12`. The `coqui-tts`
> fork fixes this and is API-compatible.

### Models

| Model | Description |
|-------|-------------|
| `tts_models/multilingual/multi-dataset/xtts_v2` | Default. 17 languages, zero-shot voice cloning. |

XTTS v2 supports 17 languages: `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`,
`ar`, `zh-cn`, `hu`, `ko`, `ja`, `hi`.

### Basic usage

```bash
# Synthesize in French (no voice cloning)
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend coqui
```

### Voice cloning with --voice-match

When `--voice-match` is enabled, anytran passes a reference audio clip from the input stream
to XTTS v2 for zero-shot voice cloning.  The synthesized speech will closely match the
speaker's voice:

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend coqui --voice-match
```

### Specifying a model

Use `--voice-model` to override the default XTTS v2 model with any coqui-tts model identifier:

```bash
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend coqui \
  --voice-model tts_models/multilingual/multi-dataset/xtts_v2
```

---

## Comparing backends

| Feature | gtts | piper | custom | fish | indextts | coqui |
|---------|------|-------|--------|------|----------|-------|
| Internet required | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Voice cloning | ✗ | ✗ | ✓ (Base) | ✓ | ✓ | ✓ |
| GPU needed | ✗ | ✗ | recommended | recommended | recommended | recommended |
| Language breadth | wide | wide | wide | wide | wide | 17 languages |
| Naturalness | medium | good | excellent | excellent | excellent | excellent |
| Speed (CPU) | fast (cloud) | fast | slow | slow | slow | slow |
| Python 3.12 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## See also

- [Voice Matching](VOICE_MATCHING.md) — auto-matching voice features for Piper, custom (Base model), fish, IndexTTS, and coqui
- [Command Line Options Reference](OPTIONS.md) — full reference for `--voice-backend`, `--voice-model`, `--voice-lang`, and `--voice-match`
- [Installation](INSTALLATION.md) — GPU support and per-feature install commands

