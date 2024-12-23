import os
import shutil
import string
import requests
from multiprocessing.pool import ThreadPool
from config import AUDIO_DIR, TEMP_DIR, DOWNLOAD_THREADS

def remove_punctuation(text):
    """Remove punctuation from a string."""
    return text.translate(str.maketrans('', '', string.punctuation))

def download_file(args):
    """Download a file from a URL and save it locally."""
    url, filename = args
    try:
        response = requests.get(url, allow_redirects=False)
        file_path = os.path.join(AUDIO_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return filename, True
    except Exception:
        return filename, False

def download_parallel(file_list):
    """Download multiple files in parallel."""
    with ThreadPool(DOWNLOAD_THREADS) as pool:
        results = pool.imap_unordered(download_file, file_list)
        for filename, success in results:
            if success:
                print(f"[\u2713] {filename}")
            else:
                print(f"[\u2715] {filename}")

def prepare_directories(overwrite=True):
    """Prepare directories for downloading and processing audio."""
    if os.path.exists(AUDIO_DIR):
        if overwrite:
            print("Deleting previously downloaded audio...")
            shutil.rmtree(AUDIO_DIR)
        else:
            print("Data already downloaded.")
            return False

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    os.makedirs(AUDIO_DIR, exist_ok=True)
    return True
