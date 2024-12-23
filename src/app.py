from downloader.download_audio_files import download_audio_files
from manifest.create_manifest import create_manifest
from resampler.resample_audio_files import resample_audio_files
from config import TEMP_DIR, AUDIO_DIR

def main():
    # Download audio files and return their URLs, filenames, and texts
    urls, filenames, texts = download_audio_files(from_cache=True)
    # Resample audio files to a uniform sampling rate
    resample_audio_files()
    # Create manifest file, metadata for each audio file with transcript and duration
    create_manifest(urls, filenames, texts)

if __name__ == "__main__":
    main()
