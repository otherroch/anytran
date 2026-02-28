# Text Translation Feature

This feature adds optional text-to-text translation capability to translate the English output from speech recognition into another language, with optional TTS voice synthesis in the target language.

## Overview

The pipeline now supports:
1. **Speech-to-Text** (via Whisper backends) - Transcribe audio to English text
2. **Text Translation** (NEW) - Translate English text to another language
3. **Text-to-Speech** (optional) - Generate voice output in the target language

## Performance

The translation backends are designed for performance:
- **googletrans**: Fast, free, no API key required
- **libretranslate**: Self-hosted, private, customizable
- **translategemma**: Local AI model, no API key, runs on GPU/CPU
- **metanllb**: Meta's NLLB local AI model, no API key, runs on GPU/CPU
- **marianmt**: Helsinki-NLP Marian MT local model, no API key, runs on GPU/CPU
- **none/passthrough**: Skip translation (default behavior)

## Usage

### Basic Text Translation

Translate speech output to Spanish:
```bash
anytran --youtube-url <URL> --youtube-api-key <KEY> \
  --output-lang es \
  --slate-text output.txt \
  --slate-backend googletrans
```

### With TTS Voice Output

Translate to French with French TTS:
```bash
anytran --rtsp rtsp://camera/stream \
  --output-lang fr \
  --slate-backend googletrans \
  --slate-voice \
  --voice-backend piper \
  --voice-model fr_FR-siwis-medium
```

### Supported Languages

Common language codes:
- `es` - Spanish
- `fr` - French  
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic
- `hi` - Hindi

## Translation Backends

### 1. Google Translate (googletrans)

**Default backend** - Fast, free, no setup required.

```bash
--slate-backend googletrans
```

**Installation:**
```bash
pip install googletrans==4.0.0-rc1
```

**Pros:**
- No API key required
- Supports 100+ languages
- Fast response times

**Cons:**
- May be rate-limited
- Requires internet connection

### 2. LibreTranslate

Self-hosted, privacy-focused translation.

```bash
--slate-backend libretranslate \
--libretranslate-url http://localhost:5000
```

**Installation:**
```bash
pip install libretranslate
libretranslate  # Run the server
```

**Pros:**
- Complete privacy (self-hosted)
- Customizable models
- No rate limits

**Cons:**
- Requires setup
- Uses more resources

### 3. TranslateGemma (Local AI)

Run translation locally using Google's TranslateGemma models (4B, 12B, or 27B parameters).

```bash
--slate-backend translategemma \
--slate-model google/translategemma-12b-it
```

**Installation:**
```bash
pip install transformers torch accelerate
```

**Model Download:**
The model will be automatically downloaded from HuggingFace on first use. Size varies by variant: ~8GB (4b), ~22GB (12b, default), ~54GB (27b).

**Pros:**
- Completely local and private
- No API keys or internet required (after download)
- High-quality translations
- No rate limits
- GPU acceleration support

**Cons:**
- Large model size (22GB+ for default)
- Requires GPU for good performance (or fast CPU)
- First run downloads the model

**Performance (12b model):**
- GPU (RTX 4090): ~1-2 seconds per sentence
- GPU (RTX 3080): ~2-4 seconds per sentence  
- CPU (16-core): ~10-30 seconds per sentence

### 4. None/Passthrough

Skip text translation entirely (default behavior).

```bash
--slate-backend none
```

### 5. MetaNLLB (Local AI)

Run translation locally using Meta's NLLB (No Language Left Behind) model.

```bash
--slate-backend metanllb \
--slate-model facebook/nllb-200-1.3B
```

**Installation:**
```bash
pip install transformers torch
```

**Model Download:**
The model will be automatically downloaded from HuggingFace on first use.

**Supported Models:**
- `facebook/nllb-200-distilled-600M` - Fastest, smallest
- `facebook/nllb-200-1.3B` - Balanced (default)
- `facebook/nllb-200-3.3B` - Higher quality, larger

**Pros:**
- Completely local and private
- No API keys or internet required (after download)
- Supports 200+ languages using FLORES-200 codes
- GPU acceleration support
- Common ISO 639-1 codes (e.g., `en`, `fr`) are automatically mapped

**Cons:**
- Requires model download (~2.5GB for 1.3B)
- Requires GPU for best performance

### 6. MarianMT (Local AI)

Run translation locally using Helsinki-NLP's Marian MT models.

```bash
--slate-backend marianmt
```

The model is **auto-derived** from the language pair as `Helsinki-NLP/opus-mt-{source}-{target}`.
For example, translating French → English automatically uses `Helsinki-NLP/opus-mt-fr-en`.
Override with `--slate-model` if you need a specific model.

**Installation:**
```bash
pip install transformers torch sentencepiece
```

Or using the package extras:
```bash
pip install -e .[marianmt]
```

**Model Download:**
The model will be automatically downloaded from HuggingFace on first use.

**Model Naming Convention:**
MarianMT models are language-pair specific and follow the pattern `Helsinki-NLP/opus-mt-{source}-{target}`.
The backend auto-selects the right model for your language pair. Common examples:

- `Helsinki-NLP/opus-mt-fr-en` - French to English (auto-selected when `--input-lang fr`)
- `Helsinki-NLP/opus-mt-en-fr` - English to French
- `Helsinki-NLP/opus-mt-en-de` - English to German
- `Helsinki-NLP/opus-mt-en-es` - English to Spanish
- `Helsinki-NLP/opus-mt-en-zh` - English to Chinese
- `Helsinki-NLP/opus-mt-en-ru` - English to Russian
- `Helsinki-NLP/opus-mt-en-ROMANCE` - English to Romance languages (fr, es, it, pt, ro)

Browse all available models at: https://huggingface.co/Helsinki-NLP

**Pros:**
- Completely local and private
- No API keys or internet required (after download)
- Lightweight models (typically 300MB or less)
- Fast inference, even on CPU
- GPU acceleration support
- Auto-selects the correct model for your language pair

**Cons:**
- Each language pair requires a separate model download
- Quality may be lower than larger models for some language pairs

## Configuration Options

### CLI Arguments

```bash
--text-translate-to LANG              # Target language code (e.g., es, fr, de)
--slate-backend BACKEND       # Translation backend (googletrans, libretranslate, translategemma, metanllb, marianmt, none)
--libretranslate-url URL              # LibreTranslate API URL (if using libretranslate)
--slate-model MODEL          # TranslateGemma model name (default: google/translategemma-12b-it)
--slate-model MODEL                # MetaNLLB model name (default: facebook/nllb-200-1.3B)
--slate-model MODEL                # MarianMT model name (default: Helsinki-NLP/opus-mt-en-ROMANCE)
--voice-lang LANG                   # Override TTS language (optional)
```

### TTS Language Override

By default, TTS uses the `--output-lang` language. Override with `--voice-lang`:

```bash
# Translate to Spanish but use English TTS
anytran --web --output-lang es --voice-lang en

# Translate to Chinese but use Japanese TTS  
anytran --web --output-lang zh --voice-lang ja
```

## Examples

### Example 1: YouTube to Spanish text file

```bash
anytran --youtube-url https://youtube.com/watch?v=... \
  --youtube-api-key YOUR_KEY \
  --output-lang es \
  --slate-text spanish_output.txt \
  --slate-backend googletrans \
  --verbose
```

### Example 2: RTSP stream with French voice output

```bash
anytran --rtsp rtsp://camera/stream \
  --output-lang fr \
  --slate-voice \
  --voice-backend piper \
  --voice-model fr_FR-siwis-medium \
  --verbose
```

### Example 3: Multi-stream with German translation

```bash
anytran --rtsp rtsp://cam1/stream --rtsp rtsp://cam2/stream \
  --output-lang de \
  --slate-text translations.txt \
  --mqtt-broker localhost
```

### Example 4: High-quality translation with timing

```bash
anytran --from-output \
  --output-lang es \
  --slate-text output.txt \
  --slate-backend libretranslate \
  --libretranslate-url http://localhost:5000 \
  --timers \
  --verbose
```

### Example 5: Local AI translation with TranslateGemma

```bash
anytran --rtsp rtsp://camera/stream \
  --output-lang fr \
  --slate-backend translategemma \
  --slate-text french_output.txt \
  --verbose
```

### Example 6: Local AI translation with MetaNLLB

```bash
anytran --rtsp rtsp://camera/stream \
  --output-lang fr \
  --slate-backend metanllb \
  --slate-text french_output.txt \
  --verbose
```

### Example 7: Local AI translation with MarianMT

```bash
# Auto-selects Helsinki-NLP/opus-mt-en-fr based on the language pair
anytran --rtsp rtsp://camera/stream \
  --output-lang fr \
  --slate-backend marianmt \
  --slate-text french_output.txt \
  --verbose
```

```bash
# Translating from French to English (auto-selects Helsinki-NLP/opus-mt-fr-en)
anytran --input slatein.txt \
  --input-lang fr \
  --slate-backend marianmt \
  --scribe-text english_output.txt \
  --verbose
```

## Performance Considerations

### Translation Speed

- **googletrans**: ~100-300ms per request
- **libretranslate**: ~50-200ms (local), ~200-500ms (remote)

### Optimization Tips

1. **Use local LibreTranslate** for lowest latency
2. **Batch short phrases** when possible
3. **Monitor timing** with `--timers` flag
4. **Choose appropriate window size** (`--window-seconds`) to balance translation frequency vs latency

### Example with Timing

```bash
anytran --youtube-url <URL> --youtube-api-key <KEY> \
  --text-translate-to es \
  --timers \
  --verbose
```

Output will show timing breakdown:
```
Timing chunk: magnitude=0.1ms, vad=5.2ms, translate_total=1234ms, text_translate=156ms, ...
```

## Integration with Existing Features

Text translation works seamlessly with:
- ✅ MQTT publishing (translated text)
- ✅ Text file output (translated text)
- ✅ Chat logging (translated text)
- ✅ TTS output (in target language)
- ✅ Multiple streams
- ✅ All whisper backends

## Troubleshooting

### Translation not working

1. **Check backend installation:**
   ```bash
   pip install googletrans==4.0.0-rc1
   ```

2. **Verify language code:**
   ```bash
   # Use 2-letter codes: es, fr, de (not "spanish", "french")
   --text-translate-to es
   ```

3. **Enable verbose mode:**
   ```bash
   --verbose
   ```

### API Key Issues

**LibreTranslate:**
```bash
# Test server connectivity
curl http://localhost:5000/languages
```

### Import Errors

If you see "Module not found" errors:
```bash
# Install all optional dependencies
pip install googletrans==4.0.0-rc1 requests
```

## Future Enhancements

Potential future additions:
- Azure Cognitive Services translator
- AWS Translate
- Google Cloud Translation API (official)
- Caching for repeated phrases
- Batch translation mode
- Custom translation models

## License

Same as parent project.
