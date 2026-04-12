# Gemma4 Setup Guide

This guide shows you how to use Google's Gemma 4 multimodal model for local audio transcription and text translation.

## What is Gemma 4?

Gemma 4 is a family of multimodal AI models from Google that can process both audio and text. In anytran, Gemma 4 can serve as:

- **Scribe backend** (`--scribe-backend gemma4`): Transcribes audio to text, replacing Whisper
- **Slate backend** (`--slate-backend gemma4`): Translates text from one language to another
- **One-pass mode**: When both scribe and slate backends are set to `gemma4` with the same model, transcription and translation happen in a single inference pass — no English pivot required

Available model sizes:
- `google/gemma-4-E4B-it` (default) — 4 billion parameters
- `google/gemma-4-E2B-it` — 2 billion parameters, faster but lower quality

## Installation

### 1. Install Dependencies

```bash
pip install transformers torch accelerate
```

This will install:
- `transformers` — HuggingFace library for loading the model
- `torch` — PyTorch deep learning framework
- `accelerate` — For efficient model loading and GPU support

### 2. GPU Support (Recommended)

Gemma 4 benefits significantly from GPU acceleration:

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Model Download (Automatic)

Models are downloaded automatically from HuggingFace on first use. Approximate sizes:
- `google/gemma-4-E2B-it` — ~5 GB
- `google/gemma-4-E4B-it` — ~9 GB

Models are cached locally after the first download.

## Usage Examples

### Audio Transcription (Scribe Backend)

Transcribe audio to English text using Gemma 4:

```bash
anytran --input audio.wav \
  --scribe-backend gemma4 \
  --scribe-text transcript.txt
```

With a specific model:

```bash
anytran --input audio.wav \
  --scribe-backend gemma4 \
  --scribe-model google/gemma-4-E2B-it \
  --scribe-text transcript.txt
```

### Text Translation (Slate Backend)

Translate text from one language to another:

```bash
anytran --input notes.txt \
  --input-lang en --output-lang fr \
  --slate-backend gemma4 \
  --slate-text french.txt
```

With a specific model:

```bash
anytran --input notes.txt \
  --input-lang en --output-lang fr \
  --slate-backend gemma4 \
  --slate-model google/gemma-4-E2B-it \
  --slate-text french.txt
```

### One-Pass Transcription + Translation

When both `--scribe-backend` and `--slate-backend` are set to `gemma4` with the same model, anytran performs transcription and translation in a single inference pass. This skips the English intermediate step entirely:

```bash
anytran --input audio.wav \
  --output-lang fr \
  --scribe-backend gemma4 \
  --slate-backend gemma4 \
  --slate-text french.txt
```

With explicit model selection for both:

```bash
anytran --input audio.wav \
  --output-lang fr \
  --scribe-backend gemma4 --scribe-model google/gemma-4-E4B-it \
  --slate-backend gemma4 --slate-model google/gemma-4-E4B-it \
  --slate-text french.txt
```

**Note:** One-pass mode activates automatically when:
- Both `--scribe-backend` and `--slate-backend` are `gemma4`
- Both use the same model (same `--scribe-model` and `--slate-model`)
- The target language is not English

If the models differ, anytran falls back to the standard two-stage pipeline (Gemma 4 transcription → Gemma 4 translation).

### RTSP Stream with Gemma 4

```bash
anytran --rtsp rtsp://camera/stream \
  --output-lang es \
  --scribe-backend gemma4 \
  --slate-backend gemma4 \
  --slate-text spanish.txt \
  --verbose
```

## Performance

### Speed Comparison

| Backend Combination | Typical Speed | Notes |
|---|---|---|
| faster-whisper + marianmt | Fast (~46s for a 5-min file) | Optimized C++ + lightweight models |
| gemma4 one-pass | Slower (~200–230s for same file) | Single model, no English pivot |
| gemma4 + gemma4 (different models) | Slowest | Two separate Gemma 4 inferences |

Gemma 4 trades speed for flexibility — a single model handles both audio understanding and translation. For latency-sensitive workloads, consider using Whisper + MarianMT or another dedicated translation backend.

### GPU vs CPU

- **GPU (CUDA)**: Recommended. Uses `bfloat16` precision and `device_map="auto"` for efficient memory usage.
- **CPU**: Supported but significantly slower. Uses `float32` precision.

## Configuration Options

| Option | Description |
|---|---|
| `--scribe-backend gemma4` | Use Gemma 4 for audio transcription |
| `--scribe-model <model>` | Gemma 4 model for transcription (default: `google/gemma-4-E4B-it`) |
| `--slate-backend gemma4` | Use Gemma 4 for text translation |
| `--slate-model <model>` | Gemma 4 model for translation (default: `google/gemma-4-E4B-it`) |

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GEMMA4_MODEL` | Default Gemma 4 model name | `google/gemma-4-E4B-it` |

## Troubleshooting

### Import Errors

If you see `Gemma4 requires transformers and torch`:
```bash
pip install transformers torch accelerate
```

### Out of Memory

If you run out of GPU memory:
- Try the smaller model: `--scribe-model google/gemma-4-E2B-it`
- Ensure no other GPU-intensive processes are running
- Fall back to CPU (slower but uses system RAM)

### Slow Performance on CPU

Gemma 4 is designed for GPU acceleration. On CPU:
- Expect 5–10× slower inference compared to GPU
- Consider using `google/gemma-4-E2B-it` for faster processing
- For production use, a GPU is strongly recommended

### Output Contains Artifacts

If you see prompt echoes, timestamps, or `[Music]` markers in the output, ensure you are running the latest version. The Gemma 4 backend includes post-processing to strip common model artifacts including:
- Prompt text echoed in the response (in English or translated to the target language)
- Translated instruction echoes (e.g., "Écoutez ceci et traduisez-le en français.")
- Timestamp markers (e.g., `[ 0m0s311ms - 0m1s211ms ]`)
- Music/sound markers (e.g., `[Music]`, `[🎵]`)
- Model apologies in English and common target languages (e.g., "I'm unable to transcribe that audio", "je suis désolé, je n'ai pas pu écouter l'audio")
- Formatting labels (e.g., `**French Translation:**`)
- Meta-instruction leaks (e.g., "Output only the translation")

### Model Quality

Gemma 4 is a general-purpose multimodal model, not a dedicated ASR system like Whisper. For the highest transcription accuracy, consider using a Whisper backend for Stage 1 and Gemma 4 only for translation:

```bash
anytran --input audio.wav \
  --output-lang fr \
  --scribe-backend faster-whisper \
  --slate-backend gemma4 \
  --slate-text french.txt
```

## Model Details

- **Architecture**: Multimodal (audio + text + image), uses `AutoModelForImageTextToText`
- **Default model**: `google/gemma-4-E4B-it` (4B parameters, instruction-tuned)
- **Smaller variant**: `google/gemma-4-E2B-it` (2B parameters, instruction-tuned)
- **License**: See model card on [HuggingFace](https://huggingface.co/google/gemma-4-E4B-it)
- **Source**: Google DeepMind
