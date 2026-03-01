import os
import argparse
import requests
import json

VOICES_BY_LANGUAGE = {
    "fr": [
        {
            "name": "fr_FR-gilles-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/gilles/low/samples/speaker_0.mp3",
        },
        {
            "name": "fr_FR-mls_1840-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/mls_1840/low/samples/speaker_0.mp3",
        },
        {
            "name": "fr_FR-mls-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/mls/low/samples/speaker_0.mp3",
        },
        {
            "name": "fr_FR-siwis-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/low/samples/speaker_0.mp3",
        },
        {
            "name": "fr_FR-tom-medium",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/tom/medium/samples/speaker_0.mp3",
        },
        {
            "name": "fr_FR-upmc-medium",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/samples/speaker_0.mp3",
        },
    ],
    "en": [
        {
            "name": "en_US-lessac-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/samples/speaker_0.mp3",
        }
    ],
    "de": [
        {
            "name": "de_DE-thorsten-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/low/samples/speaker_0.mp3",
        }
    ],
    "es": [
        {
            "name": "es_ES-mls_9972-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_9972/low/samples/speaker_0.mp3",
        }
    ],
    "it": [
        {
            "name": "it_IT-riccardo-x_low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low/samples/speaker_0.mp3",
        }
    ],
    "pt": [
        {
            "name": "pt_BR-edresson-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/samples/speaker_0.mp3",
        }
    ],
    "ru": [
        {
            "name": "ru_RU-ruslan-medium",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/samples/speaker_0.mp3",
        }
    ],
    "pl": [
        {
            "name": "pl_PL-darkman-medium",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/samples/speaker_0.mp3",
        }
    ],
    "nl": [
        {
            "name": "nl_NL-mls_5809-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/nl/nl_NL/mls_5809/low/samples/speaker_0.mp3",
        }
    ],
    "ar": [
        {
            "name": "ar_JO-kareem-low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/low/samples/speaker_0.mp3",
        }
    ],
    "zh": [
        {
            "name": "zh_CN-huayan-x_low",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/samples/speaker_0.mp3",
        }
    ],
}


def download_mp3(url, filename):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        default="fr",
        help="Comma-separated language codes to process (default: fr). Use 'all' for every configured language.",
    )
    return parser.parse_args()


def get_selected_languages(raw_languages):
    requested_languages = [lang.strip().lower() for lang in raw_languages.split(",") if lang.strip()]
    if not requested_languages:
        return ["fr"]
    if requested_languages == ["all"]:
        return list(VOICES_BY_LANGUAGE.keys())
    return requested_languages


def run(selected_languages):
    import librosa
    from anytran.voice_matcher import extract_voice_features

    voices = []
    for language_code in selected_languages:
        voices.extend(VOICES_BY_LANGUAGE.get(language_code, []))

    results = []
    os.makedirs("voice_samples", exist_ok=True)
    for v in voices:
        mp3_file = f"voice_samples/{v['name']}.mp3"
        if not os.path.exists(mp3_file):
            try:
                download_mp3(v["url"], mp3_file)
            except Exception as e:
                print(f"Error downloading {v['name']}: {e}")
                results.append({"voice": v["name"], "pitch": 0, "gender": "download_error"})
                continue
        try:
            y, sr = librosa.load(mp3_file, sr=None)
        except Exception as e:
            print(f"Error processing audio for {v['name']}: {e}")
            continue

        features = extract_voice_features(y, sr)
        results.append(
            {
                v["name"]: {
                    "pitch": round(features["mean_pitch"], 2),
                    "gender": features["gender"],
                }
            }
        )

    output_file = f"{selected_languages[0]}_voice_table.json" if len(selected_languages) == 1 else "voice_table.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_args()
    run(get_selected_languages(args.languages))
