# TranslateGemma Setup Guide

This guide shows you how to use Google's TranslateGemma model for local, offline translation.

## What is TranslateGemma?

TranslateGemma is a family of powerful AI models from Google that run locally on your machine. They provide high-quality translations without needing internet connectivity or API keys (after initial model download). Available sizes include 4B, 12B, and 27B parameters. The default model is `google/translategemma-12b-it`.

## Installation

### 1. Install Dependencies

```bash
pip install -e .[translategemma]
```

This will install:
- `transformers` - HuggingFace library for loading the model
- `torch` - PyTorch deep learning framework
- `accelerate` - For efficient model loading

### 2. Download the Model (First Run)

The model will be automatically downloaded on first use. The download size depends on the model variant:
- `google/translategemma-4b-it` — approximately 8GB
- `google/translategemma-12b-it` (default) — approximately 22GB
- `google/translategemma-27b-it` — approximately 54GB

Ensure you have:
- Sufficient disk space (~25GB free recommended for the default 12B model)
- A good internet connection for the initial download
- Patience (download may take 15–60 minutes depending on your connection and chosen model)

The model is cached locally, so subsequent runs won't require downloading.

## Usage Examples

### Basic Translation

Translate RTSP stream to French:

```bash
anytran --rtsp rtsp://camera/stream \
  --output-lang fr \
  --slate-backend translategemma \
  --slate-text french_output.txt \
  --verbose
```

### Custom Model

Use a different TranslateGemma model variant:

```bash
anytran --rtsp rtsp://camera/stream \
  --output-lang es \
  --slate-backend translategemma \
  --slate-model google/translategemma-12b-it \
  --slate-text spanish.txt
```

### With Voice Output

Translate to German with TTS:

```bash
anytran --from-output \
  --output-lang de \
  --slate-backend translategemma \
  --slate-voice \
  --voice-backend piper \
  --voice-model de_DE-thorsten-medium
```

### YouTube Translation

```bash
anytran --youtube-url https://youtube.com/watch?v=VIDEO_ID \
  --youtube-api-key YOUR_KEY \
  --output-lang ja \
  --slate-backend translategemma \
  --slate-text japanese.txt
```

## Performance

Translation speed depends on your hardware:

| Hardware | Approximate Speed |
|----------|------------------|
| RTX 4090 | 1-2 seconds/sentence |
| RTX 3080 | 2-4 seconds/sentence |
| RTX 3060 | 4-6 seconds/sentence |
| 16-core CPU | 10-30 seconds/sentence |
| 8-core CPU | 20-60 seconds/sentence |

### GPU Acceleration

For best performance, ensure PyTorch can use your GPU:

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not available but you have an NVIDIA GPU:
1. Install CUDA Toolkit from NVIDIA
2. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

## Supported Languages

TranslateGemma supports translation between many language pairs. Common ones include:

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Hindi (hi)
- Dutch (nl)
- Polish (pl)
- Turkish (tr)
- Swedish (sv)
- Danish (da)
- Norwegian (no)
- Finnish (fi)

## Troubleshooting

### Model Download Fails

If the download fails or times out:

1. Check your internet connection
2. Try downloading manually:
   ```python
   from transformers import AutoProcessor, AutoModelForImageTextToText
   model = AutoModelForImageTextToText.from_pretrained("google/translategemma-12b-it")
   processor = AutoProcessor.from_pretrained("google/translategemma-12b-it")
   ```

### Out of Memory Error

If you get GPU out of memory errors:

1. Try using CPU mode (slower):
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

2. Close other GPU-intensive applications

3. Reduce batch size or use smaller window sizes:
   ```bash
   anytran --rtsp ... --window-seconds 3.0
   ```

### Slow Performance on CPU

If translation is too slow on CPU:

1. Consider using a faster backend like `googletrans` or `libretranslate`
2. Reduce the frequency of translations with larger window sizes
3. Use GPU if available

### Import Errors

If you see "Module not found" errors:

```bash
pip install transformers torch accelerate
```

## Comparison with Other Backends

| Backend | Speed | Quality | Privacy | Cost | Internet |
|---------|-------|---------|---------|------|----------|
| googletrans | Fast | Good | No | Free | Required |
| libretranslate | Medium | Good | Yes | Free | Optional |
| **translategemma** | **Slow-Medium** | **Excellent** | **Yes** | **Free** | **No*** |

\* Internet required only for initial model download

## Privacy Benefits

TranslateGemma runs entirely on your local machine:
- No data sent to external servers
- No API keys or accounts needed
- Complete privacy for sensitive content
- Works offline (after initial setup)

Perfect for:
- Confidential business communications
- Medical/legal transcriptions
- Air-gapped environments
- Scenarios requiring data sovereignty

## Tips for Best Results

1. **Use verbose mode** to see translation progress:
   ```bash
   --verbose
   ```

2. **Monitor timing** to understand performance:
   ```bash
   --timers
   ```

3. **Start with shorter window sizes** for faster feedback:
   ```bash
   --window-seconds 3.0
   ```

4. **Warm up the model** - first translation is slower as the model loads

5. **Use GPU** whenever possible for 5-10x speed improvement

## Model Details

- **Default model**: google/translategemma-12b-it
- **Available models**: google/translategemma-4b-it (8GB), google/translategemma-12b-it (22GB), google/translategemma-27b-it (54GB)
- **Architecture**: Gemma 2 (Google's open-source LLM family)
- **License**: Google AI Gemma Terms of Use
- **Source**: https://huggingface.co/google/translategemma-12b-it

## Additional Resources

- [TranslateGemma on HuggingFace](https://huggingface.co/google/translategemma-12b-it)
- [Text Translation Documentation](TEXT_TRANSLATION.md)
- [Command Line Options](OPTIONS.md)
