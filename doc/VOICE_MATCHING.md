# Voice Matching Feature

This document describes the voice matching feature added to voicetran.

## Overview

The voice matching feature allows the output voice (slate-voice) to match the input voice characteristics:

1. **`--voice-match`**: Automatically selects the best matching Piper TTS voice based on the input speaker's characteristics (pitch, gender, etc.)

For backends that support voice cloning (`custom`, `fish`, `indextts`, `coqui`), `--voice-match` additionally supplies a reference audio clip from the input stream to enable zero-shot voice cloning.

## Installation

### For Voice Matching

Voice matching is included in the base installation and works with Piper TTS:

```bash
pip install -e .[piper]
```

## Usage

### Auto Voice Matching (--voice-match)

This feature analyzes the input audio to extract voice characteristics (pitch, gender, brightness) and automatically selects the closest matching Piper voice model.

```bash
# Basic usage with voice matching
anytran --input audio.mp3 \
  --output-lang es \
  --slate-voice output.wav \
  --voice-backend piper \
  --voice-match

# Real-time RTSP with voice matching
anytran --rtsp rtsp://camera-url \
  --output-lang fr \
  --slate-voice french.wav \
  --voice-backend piper \
  --voice-match \
  --verbose

# YouTube with voice matching
anytran --youtube-url "https://youtube.com/watch?v=..." \
  --youtube-api-key YOUR_API_KEY \
  --output-lang de \
  --slate-text german.txt \
  --voice-backend piper \
  --voice-match
```

**How it works:**
1. Extracts fundamental frequency (pitch) from input audio
2. Estimates gender based on pitch range
3. Analyzes spectral characteristics (brightness/timbre)
4. Selects the closest matching Piper voice for the target language
5. Falls back to default voice if no suitable match is found

**Supported Piper Voices:**
- English: libritts-high, lessac-medium, amy-medium (female), ryan-high, norman-medium, joe-medium (male)
- French: siwis-medium (female), upmc-medium (male)
- Spanish: mls_10246-low (female), carlfm-x_low (male)
- German: eva_k-x_low (female), thorsten-high (male)

### Voice Cloning with coqui (--voice-backend coqui --voice-match)

When `--voice-match` is used with the `coqui` backend, anytran supplies a reference audio clip
from the input stream to XTTS v2 for zero-shot voice cloning:

```bash
# Zero-shot voice cloning with coqui-tts XTTS v2
anytran --input sample.wav --output-lang fr --slate-voice output.wav \
  --voice-backend coqui --voice-match

# Real-time RTSP with coqui voice cloning
anytran --rtsp rtsp://camera-url \
  --output-lang fr \
  --slate-voice french.wav \
  --voice-backend coqui \
  --voice-match \
  --verbose
```

**Supported languages for coqui XTTS v2:**
`en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh-cn`, `hu`, `ko`, `ja`, `hi`

Install: `pip install "anytran[coqui]"`

## Combining with Other Features

The voice matching feature works seamlessly with other voicetran features:

```bash
# With text translation
anytran --input english.mp3 \
  --output-lang fr \
  --slate-text french.txt \
  --slate-voice french.wav \
  --voice-match

# With web interface
anytran --web \
  --web-port 8443 \
  --web-ssl-self-signed \
  --output-lang es \
  --voice-match \
  --verbose

# With MQTT output
anytran --rtsp rtsp://camera1 \
  --mqtt-broker localhost \
  --mqtt-topic camera1/translation \
  --output-lang de \
  --voice-match
```

## Regenerating the Voice Table

The voice table JSON file (`src/anytran/voice_table.json`) stores pre-computed voice features for available Piper TTS voices. You can regenerate it using the built-in `--voice-table-gen` subcommand:

```bash
# Regenerate for French only (default)
anytran --voice-table-gen

# Regenerate for French and English
anytran --voice-table-gen --voice-table-lang fr,en

# Regenerate for all supported languages
anytran --voice-table-gen --voice-table-lang all

# Write to a custom path
anytran --voice-table-gen --voice-table-lang fr,en --voice-table-output ./my_voice_table.json
```

**When to regenerate:**
- After installing new Piper voice models
- When you need to add support for additional languages
- If the voice table file becomes stale or corrupted

See [OPTIONS.md](OPTIONS.md) for the full reference on `--voice-table-gen`, `--voice-table-lang`, and `--voice-table-output`.

## Technical Details

### Voice Feature Extraction

The `extract_voice_features()` function in `voice_matcher.py` extracts:

- **Mean Pitch**: Fundamental frequency (F0) using YIN algorithm
- **Pitch Variation**: Standard deviation of pitch
- **Zero Crossing Rate**: Proxy for speaking rate and energy
- **Spectral Centroid**: Brightness/timbre of the voice
- **Gender Estimation**: Based on pitch ranges (Male: <140Hz, Female: >180Hz)
- **Voice Type**: Categorized as male_deep, male_mid, female_low, female_mid, female_high

### Voice Selection Algorithm

The `select_best_piper_voice()` function:

1. Normalizes the target language code
2. Filters available voices by target language
3. Prioritizes gender match
4. Finds minimum pitch difference within gender
5. Falls back to closest pitch if no gender match

## Performance Considerations

### --voice-match
- Minimal overhead (~10-50ms per audio chunk for feature extraction)
- No additional memory requirements
- Works well for real-time applications

## Troubleshooting

### Voice Matching Issues

**No suitable voice found:**
- Ensure Piper voices for the target language are installed
- Check verbose output for feature extraction results
- Try different Piper voice models manually with `--voice-model`

**Poor voice match:**
- Input audio quality may be low (background noise, distortion)
- Consider using longer audio segments (`--window-seconds 10`)

## Examples

See `examples/` directory for complete examples:
- `voice_match_example.sh`: Basic voice matching
- `rtsp_voice_match.sh`: Real-time RTSP with voice matching

## Credits

- Voice feature extraction uses [librosa](https://librosa.org/)
- Piper TTS voices from [Rhasspy Piper](https://github.com/rhasspy/piper)

## Future Enhancements

Potential improvements being considered:

- Pre-compute voice embeddings for faster matching
- Support for preserving speaking rate/tempo
- Multi-speaker detection and tracking
- Custom voice training from user samples
- Voice conversion (change voice without translation)
