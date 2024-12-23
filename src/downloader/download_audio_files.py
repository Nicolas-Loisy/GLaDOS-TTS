import os
import requests
from bs4 import BeautifulSoup
from utils.utils import prepare_directories, download_parallel, remove_punctuation
from config import AUDIO_DIR, sources, blocklist

def download_audio_files():
    urls, filenames, texts = [], [], []

    # Extract audio links and texts from sources
    for source in sources:
        response = requests.get(source, allow_redirects=False)
        soup = BeautifulSoup(response.text, 'html.parser')

        for link in soup.find_all('a', href=True):
            url = link['href']
            if url.endswith(".wav") and "https:" in url:
                list_item = link.find_parent("li")
                text = list_item.find('i').get_text(strip=True) if list_item else ""
                filename = url.split("/")[-1]

                if all(s not in url for s in blocklist) and text:
                    urls.append(url)
                    filenames.append(filename)
                    texts.append(remove_punctuation(text))

    print(f"Found {len(urls)} audio files.")

    # Prepare directories and download files
    if prepare_directories():
        download_parallel(zip(urls, filenames))

    return urls, filenames, texts
