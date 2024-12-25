from downloader.download_audio_files import download_audio_files
from manifest.manage_manifest import create_manifest, split_manifest
from resampler.resample_audio_files import resample_audio_files
from utils.fastpitch_training import load_fastpitch_model, extract_pitch_statistics, train_fastpitch

def main():
    # Download audio files and return their URLs, filenames, and texts
    urls, filenames, texts = download_audio_files(from_cache=True)
    # Resample audio files to a uniform sampling rate
    resample_audio_files()
    # Create manifest file, metadata for each audio file with transcript and duration
    create_manifest(urls, filenames, texts)
    
    split_manifest()

    # Charger le modèle
    model = load_fastpitch_model()
    # Extraire les statistiques de pitch
    train_manifest = "./audio/manifest_train.json"
    val_manifest = "./audio/manifest_validation.json"
    pitch_mean, pitch_std, pitch_min, pitch_max = extract_pitch_statistics(train_manifest)
    
    # Entraîner le modèle
    train_fastpitch(
        model,
        train_manifest,
        val_manifest,
        pitch_mean,
        pitch_std,
        pitch_min,
        pitch_max,
        num_epochs=10,
        lr=1e-4
    )
if __name__ == "__main__":
    main()
