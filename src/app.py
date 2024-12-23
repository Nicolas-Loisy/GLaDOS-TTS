from downloader.download_audio_files import download_audio_files
from manifest.create_manifest import create_manifest
from resampler.resample_audio_files import resample_audio_files
from config import TEMP_DIR, AUDIO_DIR

def main():
    urls, filenames, texts = download_audio_files()
    create_manifest(urls, filenames, texts)
    resample_audio_files()

if __name__ == "__main__":
    main()
