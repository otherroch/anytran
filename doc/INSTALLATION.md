# Installation

## Linux system prerequisites

On Linux, you may need to install the following system packages before installing Python dependencies (as used in the [dockerfile](../dockerfile)):

```bash
sudo apt-get install -y portaudio19-dev build-essential ffmpeg git cmake
```

## GPU support

- you may need to install the appropriate torch version (see https://pytorch.org/get-started/locally/)
- for example:

   ` pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130`
## Development environment

- `pip install --group dev --group all -e .[all]`

## Docker images 

- Built with `Dockerfile`
- `docker pull otherroch/anytran`
- `docker run -it --gpus all otherroch/anytran --help`  

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
- Whisper backends:
  - `pip install -e .[whispercpp]` or
  - `pip install -e .[faster-whisper]` or
  - `pip install -e .[whisper-ctranslate2]`
- Windows output capture: `pip install -e .[output]`
