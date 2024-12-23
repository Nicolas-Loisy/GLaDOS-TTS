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
