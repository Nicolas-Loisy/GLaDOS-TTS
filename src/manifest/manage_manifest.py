import os
import json
import re
import num2words
from resampler.resample_audio_files import audio_duration
from config import AUDIO_DIR

def create_manifest(urls, filenames, texts):
    """Create a manifest file with audio metadata."""
    manifest_path = os.path.join(AUDIO_DIR, "manifest.json")
    print(f"Creating manifest file: {manifest_path}")

    total_audio_time = 0

    manifest_data = []

    for url, filename, text in zip(urls, filenames, texts):
        file_path = os.path.join(AUDIO_DIR, filename)
        duration = audio_duration(file_path)

        item = {
            "audio_filepath": file_path,
            "text": re.sub(r"\d+", lambda x: num2words.num2words(int(x.group())), text).lower(),
            "duration": duration
        }

        total_audio_time += duration
        manifest_data.append(item)

    with open(manifest_path, 'w') as manifest:
        json.dump(manifest_data, manifest, indent=4)

    print(f"Total audio duration: {total_audio_time / 60:.2f} minutes.")

def read_manifest():
    with open(f"{AUDIO_DIR}/manifest.json", 'r') as f:
        manifest = json.load(f)
    return manifest

def split_manifest():
    manifest = read_manifest()

    validation_data = manifest[-5:]
    train_data = manifest[:-5]

    with open(f"{AUDIO_DIR}/manifest_validation.json", 'w') as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=4)

    with open(f"{AUDIO_DIR}/manifest_train.json", 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
