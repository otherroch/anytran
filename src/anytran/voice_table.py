import os
import argparse
import requests
import json
from requests import RequestException

VOICES_JSON_URLS = (
    "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json",
    "https://raw.githubusercontent.com/LouisGameDev/piper-voices/main/voices.json",
)
VOICES_RESOLVE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
DEFAULT_OUTPUT_FILE = "src/anytran/voice_table.json"


def download_mp3(url, filename):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)


def load_voices_catalog():
    errors = []
    for url in VOICES_JSON_URLS:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except (RequestException, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not fetch Piper voices catalog:\n" + "\n".join(errors))


def collect_voices_for_languages(catalog, selected_languages):
    selected = {lang.lower() for lang in selected_languages}
    all_families = {
        voice.get("language", {}).get("family", "").lower()
        for voice in catalog.values()
        if voice.get("language", {}).get("family")
    }
    all_codes = {
        voice.get("language", {}).get("code", "").lower()
        for voice in catalog.values()
        if voice.get("language", {}).get("code")
    }
    if selected == {"all"}:
        selected = all_families

    voices = []
    for voice in catalog.values():
        language = voice.get("language", {})
        family = language.get("family", "").lower()
        code = language.get("code", "").lower()
        if family not in selected and code not in selected:
            continue
        for file_path in voice.get("files", {}):
            if not file_path.endswith(".onnx"):
                continue
            voice_dir = os.path.dirname(file_path)
            onnx_file = os.path.basename(file_path)
            voices.append(
                {
                    "onnx_file": onnx_file,
                    "name": os.path.splitext(onnx_file)[0],
                    "url": f"{VOICES_RESOLVE_BASE_URL}/{voice_dir}/samples/speaker_0.mp3",
                }
            )
            break

    unknown_languages = sorted(
        requested
        for requested in selected
        if requested not in all_families
        and requested not in all_codes
    )
    return voices, unknown_languages


def load_existing_entries(output_file):
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, list) else []
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def append_unique_entries(output_file, new_entries):
    existing_entries = load_existing_entries(output_file)
    existing_by_onnx = {
        entry.get("onnx_file"): entry
        for entry in existing_entries
        if isinstance(entry, dict) and entry.get("onnx_file")
    }
    for entry in new_entries:
        onnx_file = entry.get("onnx_file")
        if onnx_file and onnx_file not in existing_by_onnx:
            existing_entries.append(entry)
            existing_by_onnx[onnx_file] = entry
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_entries, f, indent=2, ensure_ascii=False)
    return existing_entries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        "--languages",
        dest="language",
        default="fr",
        help="Comma-separated language codes to process (default: fr). Use 'all' for every configured language.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path. Existing files are appended with new unique entries.",
    )
    return parser.parse_args()


def get_selected_languages(raw_languages):
    requested_languages = [lang.strip().lower() for lang in raw_languages.split(",") if lang.strip()]
    if not requested_languages:
        return ["fr"]
    if requested_languages == ["all"]:
        return ["all"]
    return requested_languages


def run(selected_languages, output_file):
    import librosa
    from .voice_matcher import extract_voice_features

    if not selected_languages:
        selected_languages = ["fr"]

    catalog = load_voices_catalog()
    voices, unknown_languages = collect_voices_for_languages(catalog, selected_languages)

    if unknown_languages:
        print(f"Warning: unsupported language code(s): {', '.join(unknown_languages)}")

    results = []
    os.makedirs("voice_samples", exist_ok=True)
    for v in voices:
        mp3_file = f"voice_samples/{v['name']}.mp3"
        if not os.path.exists(mp3_file):
            try:
                download_mp3(v["url"], mp3_file)
            except Exception as e:
                print(f"Error downloading {v['name']}: {e}")
                continue
        try:
            y, sr = librosa.load(mp3_file, sr=None)
        except Exception as e:
            print(f"Error processing audio for {v['name']}: {e}")
            continue

        features = extract_voice_features(y, sr)
        results.append(
            {
                "onnx_file": v["onnx_file"],
                "pitch": round(features["mean_pitch"], 2),
                "pitch_std": round(features["pitch_std"], 2),
                "zcr": round(features["zcr"], 4),
                "brightness": round(features["brightness"], 2),
                "gender": features["gender"],
            }
        )

    saved_results = append_unique_entries(output_file, results)

    print(json.dumps(saved_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_args()
    run(get_selected_languages(args.language), args.output)
