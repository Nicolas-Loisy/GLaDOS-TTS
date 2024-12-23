import os
import requests
import json
from bs4 import BeautifulSoup
from utils.utils import prepare_directories, download_parallel, remove_punctuation
from config import CACHE_FILE, sources, blocklist
from tqdm import tqdm

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def download_audio_files(from_cache=False):
    urls, filenames, texts = [], [], []
    cache = load_cache() if from_cache else {}

    # Extract audio links and texts from sources
    for source in sources:
        if source in cache:
            print(f"Using cached data for {source}")
            source_data = cache[source]
        else:
            print(f"Extracting audio links from {source}")
            response = requests.get(source, allow_redirects=False)
            print(f"Response: ok={response.ok}, status_code={response.status_code}, url={response.url}")
            soup = BeautifulSoup(response.text, 'html.parser')
            source_data = []

            for link in tqdm(soup.find_all('a', href=True)):
                url = link['href']
                if url.endswith(".wav") and "https:" in url:
                    list_item = link.find_parent("li")
                    text = link.get_text(strip=True) if list_item else ""
                    filename = url.split("/")[-1]

                    if all(s not in url for s in blocklist) and text:
                        source_data.append((url, filename, remove_punctuation(text)))

            cache[source] = source_data
            save_cache(cache)

        for url, filename, text in source_data:
            urls.append(url)
            filenames.append(filename)
            texts.append(text)

    print(f"Found {len(urls)} audio files.")

    # Prepare directories and download files
    overwrite=False if from_cache else True
    if prepare_directories(overwrite=overwrite):
        download_parallel(zip(urls, filenames))

    return urls, filenames, texts
