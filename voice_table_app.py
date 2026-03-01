import os
import requests
import librosa
import numpy as np
import json
import soundfile as sf
from anytran.voice_matcher import extract_voice_features

# List of FR voices and their sample URLs
voices = [
    {
        "name": "fr_FR-gilles-low",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/gilles/low/samples/speaker_0.mp3"
    },
    {
        "name": "fr_FR-mls_1840-low",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/mls_1840/low/samples/speaker_0.mp3"
    },
    {
        "name": "fr_FR-mls-low",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/mls/low/samples/speaker_0.mp3"
    },
    {
        "name": "fr_FR-siwis-low",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/low/samples/speaker_0.mp3"
    },
    {
        "name": "fr_FR-tom-medium",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/tom/medium/samples/speaker_0.mp3"
    },
    {
        "name": "fr_FR-upmc-medium",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/samples/speaker_0.mp3"
    }
]

def download_mp3(url, filename):
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)



results = []
os.makedirs("voice_samples", exist_ok=True)
for v in voices:
    mp3_file = f"voice_samples/{v['name']}.mp3"
    if not os.path.exists(mp3_file):
        try:
            download_mp3(v["url"], mp3_file)
        except Exception as e:
            print(f"Error downloading {v['name']}: {e}")
            results.append({
                "voice": v["name"],
                "pitch": 0,
                "gender": "download_error"
            })
            continue
    try:
       y, sr = librosa.load(mp3_file, sr=None)
    except Exception as e:
        print(f"Error processing audio for {v['name']}: {e}")
        continue
    
    features = extract_voice_features(y, sr)
    results.append(
        {
           v["name"] :
              {
                "pitch" : round(features["mean_pitch"], 2),
                "gender": features["gender"]
              }
        }
    )

with open("fr_voice_table.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(json.dumps(results, indent=2, ensure_ascii=False))