## Quick start
- Clone from GitHub
  - `git clone https://github.com/otherroch/anytran.git`

- `cd anytran`

- Create python 3.12 environment using venv
  - `python3.12 -m venv .venv`  on Linux/Mac 
  
  OR 
  
  - `py -3.12 -m venv .venv` on Windows

  
- alternatively create python 3.12 environment using conda

  - `conda create -n anytran python=3.12` 
     
     (specify a name and python version)

- Activate the environment
  - `source .venv/bin/activate` or `. .venv/bin/activate` on Linux/Mac
  - `.venv\Scripts\activate` on Windows

    OR

  - `conda activate anytran`

- Install all features
  - `pip install --group all -e .[all]`
    (may require `pip install -U pip`)

- For individual feature installs, GPU support, or Linux system prerequisites, see [INSTALLATION.md](doc/INSTALLATION.md)

## Run

Examples:
- RTSP:
  - `anytran --rtsp rtsp://... --scribe-text transcript.txt`
- System output (Windows):
  - `anytran --from-output --scribe-text transcript.txt`
- Web server:
  - `anytran --web --scribe-text transcript.txt`
- YouTube:
  - `anytran --youtube-url https://... --youtube-api-key YOUR_KEY --scribe-text transcript.txt`
- Translate to French (text):
  - `anytran --rtsp rtsp://... --output-lang fr --slate-text french.txt`
- Translate to French (voice):
  - `anytran --from-output --output-lang fr --slate-voice`
- File input (audio):
  - `anytran --input sample.wav --scribe-text transcript.txt`
- File input (audio -> translate):
  - `anytran --input sample.wav --output-lang fr --slate-text french.txt`
- File input (text -> translate):
  - `anytran --input notes.txt --input-lang en --output-lang es --slate-text spanish.txt`
- File input (text -> translate, local AI backend):
  - `anytran --input notes.txt --input-lang en --output-lang fr --slate-text french.txt --slate-backend marianmt`
- Generate voice table (for voice matching):
  - `anytran --voice-table-gen --voice-table-lang fr,en`

## Timing summaries
- `anytran --input sample.wav --timers`  # Print timing summary by stage
- `anytran --input sample.wav --timers-all`  # Print all timing summaries (full, by stage, overhead)

## Documentation

For a complete reference of all command line options and how they interact with each other, see:
- [Command Line Options Reference](doc/OPTIONS.md)

Additional topic guides:
- [Installation](doc/INSTALLATION.md) — CUDA GPU support, individual feature installs, and Linux system prerequisites
- [Text Translation Backends](doc/TEXT_TRANSLATION.md) — details on all translation backends (googletrans, libretranslate, MarianMT, MetaNLLB, TranslateGemma)
- [TranslateGemma Setup](doc/TRANSLATEGEMMA_SETUP.md) — guide for using Google's local TranslateGemma AI model
- [TTS Backends](doc/TTS_BACKENDS.md) — details on all voice synthesis backends (gtts, piper, custom/Qwen3-TTS, fish-speech, IndexTTS)
- [Voice Matching](doc/VOICE_MATCHING.md) — auto-matching voice features
- [Loop Translation (looptran)](doc/LOOPTRAN.md) — iterative back-translation with `--looptran` and `--tran-converge`

## License

This project is licensed under the [Apache License 2.0](LICENSE).
