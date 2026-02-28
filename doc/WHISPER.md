# Whisper Backend Configuration

This document describes configuration options specific to whisper.cpp and whisper-ctranslate2 backends.

---

## Scribe Options: whisper.cpp-Specific Configuration

These options only apply when `--scribe-backend whispercpp` or `--scribe-backend whispercpp-cli` is selected.

### `--whispercpp-bin <path>`
- **Type**: File path
- **Environment variable**: `WHISPERCPP_BIN`
- **Default**: 
  - Windows: `$HOME/whisper.cpp/build/bin/Release/whisper-cli.exe`
  - Linux/Mac: `$HOME/whisper.cpp/build/bin/whisper-cli`
- **Purpose**: Path to the compiled whisper.cpp binary
- **Interactions**:
  - Required for `--scribe-backend whispercpp` or `--scribe-backend whispercpp-cli`
  - `--verbose` will show the configured path
  - Must be built before use (not downloaded automatically)

### `--whispercpp-model-dir <path>`
- **Type**: Directory path
- **Environment variable**: `WHISPERCPP_MODEL_DIR`
- **Default**: `./models`
- **Purpose**: Directory where whisper.cpp models are stored
- **Behavior**: 
  - Can be relative (expanded from project root) or absolute path
  - Auto-created if doesn't exist and auto-download is enabled
- **Interactions**:
  - `--scribe-model`: Looks for `ggml-{model-name}.bin` in this directory
  - `--auto-download` (enabled by default): Downloads models here if missing
  - Affects `--no-auto-download` behavior

### `--whispercpp-threads <int>`
- **Type**: Integer (≥ 1)
- **Environment variable**: `WHISPERCPP_THREADS`
- **Default**: `4`
- **Purpose**: Number of CPU threads for whisper.cpp inference
- **Performance tuning**:
  - Higher = faster (up to number of available cores)
  - Recommended: Set to physical (not logical) core count
  - Too many threads may reduce performance due to contention
- **Interactions**:
  - Only applies to `--scribe-backend whispercpp`
  - No effect on other backends

### `--auto-download` (deprecated) vs `--no-auto-download`
- **Type**: Boolean flag
- **Default behavior**: Auto-download is **enabled** by default
- **Behavior**:
  - When enabled: Automatically downloads missing whisper.cpp models from HuggingFace
  - When disabled: Fails with error if model not found
- **Recommended**: Leave auto-download enabled for convenience
- **Interactions**:
  - Only applies to `--scribe-backend whispercpp`
  - Uses `--whispercpp-model-dir` as download destination
  - Requires internet connection when enabled
  - `--verbose` shows download progress

### `--whispercpp-cli-detect-lang`
- **Type**: Boolean flag
- **Default**: Disabled
- **Purpose**: Use whisper.cpp CLI for initial language detection
- **When to use**: When automatic language detection is unreliable
- **Effect**: May increase processing time due to extra language detection pass
- **Interactions**:
  - Only applies to `--scribe-backend whispercpp` or `--backend whispercpp-cli`
  - Works with `--input-lang auto`

---

## Scribe Options: whisper-ctranslate2-Specific Configuration

These options only apply when `--scribe-backend whisper-ctranslate2` is selected.

### `--whisper-ctranslate2-device <device>`
- **Type**: Choice
- **Choices**: `auto`, `cuda`, `cpu`
- **Default**: `auto`
- **Environment variable**: `WHISPER_CTRANSLATE2_DEVICE`
- **Command-line precedence**: Overrides environment variable if both are set
- **Behavior**:
  - `auto`: Automatically selects CUDA if available, falls back to CPU
  - `cuda`: Forces GPU acceleration (requires NVIDIA GPU and CUDA libraries)
  - `cpu`: Forces CPU-only processing
- **Performance**:
  - GPU (CUDA): Much faster for large models
  - CPU: Works everywhere but slower
- **Interactions**:
  - `--whisper-ctranslate2-device-index`: Selects which GPU (if using CUDA)
  - `--whisper-ctranslate2-compute-type`: Sets precision for the chosen device

### `--whisper-ctranslate2-device-index <int>`
- **Type**: Integer ≥ 0
- **Environment variable**: `WHISPER_CTRANSLATE2_DEVICE_INDEX`
- **Default**: Not set (uses device 0)
- **Purpose**: Selects which GPU to use (if multiple GPUs available)
- **Behavior**: Only applicable when `--whisper-ctranslate2-device cuda` is used
- **Example**: `--whisper-ctranslate2-device-index 1` uses second GPU
- **Interactions**:
  - Only relevant when device is `cuda`
  - Ignored if using CPU device

### `--whisper-ctranslate2-compute-type <type>`
- **Type**: String
- **Environment variable**: `WHISPER_CTRANSLATE2_COMPUTE_TYPE`
- **Default**: `default`
- **Common values**: `default`, `float16`, `int8`, `int16`
- **Purpose**: Controls numerical precision and memory usage
- **Options**:
  - `default`: Auto-selected based on device capability
  - `float16`: Half precision (faster, less memory, lower accuracy)
  - `int8`: Quantized (very fast, very low memory)
  - `int16`: 16-bit integer quantization
- **GPUs**: Typically support `float16` for significant speedup
- **CPUs**: May not support all types; `default` usually best
- **Interactions**:
  - Only applies to `--scribe-backend whisper-ctranslate2`
  - Works with both `cuda` and `cpu` devices
  - Affects speed/memory tradeoff
