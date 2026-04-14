# Installation

## Linux system prerequisites

On Linux, you may need to execute this shell script to install various system packages **before** installing Python dependencies:

`sudo ./install_pkg.sh` (script: [`install_pkg.sh`](../install_pkg.sh), as used in the [Dockerfile](../Dockerfile))

## GPU support

- you may need to install the appropriate torch version (see https://pytorch.org/get-started/locally/)
- for example:

   ` pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130`
## Development environment

- `pip install --group dev --group all -e .`

## Docker images 

- Built with [Dockerfile](../Dockerfile)
- `docker pull otherroch/anytran`
- `docker run -it --rm --gpus all otherroch/anytran --help`  

## Install pywhispercpp on CUDA GPU (optional feature)

- `./src/anytran/pywhispercppcuda.sh` on Linux

   OR

- `src/anytran/CUDApywhispercpp.ps1` on Windows

- otherwise CPU version will be installed by default (below)

## Install base dependencies

- `pip install -e .`

## Plus Optional features

- RTSP: `pip install -e .[rtsp]`
- Web: `pip install -e .[web]`
- YouTube: `pip install -e .[youtube]`
- VAD: `pip install -e .[vad]`
- Piper TTS: `pip install -e .[piper]`
- Custom TTS (Qwen3-TTS): `pip install -e .[custom]`
- Fish-speech TTS: `pip install -e .[fish]`
- Whisper backends:
  - `pip install -e .[whispercpp]` or
  - `pip install -e .[faster-whisper]` or
  - `pip install -e .[whisper-ctranslate2]`
- Windows output capture: `pip install -e .[output]`
- IndexTTS:
  ```bash
  GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/index-tts/index-tts.git
  pip install "anytran[index-tts]"
  ```
  > Note: IndexTTS is not published on PyPI. The `GIT_LFS_SKIP_SMUDGE=1` flag skips large example audio files from Git LFS that are not needed at runtime.
