from downloader.download_audio_files import download_audio_files
from manifest.manage_manifest import create_manifest, split_manifest
from resampler.resample_audio_files import resample_audio_files
from utils.fastpitch_training import (
    load_models,
    load_text_processing,
    extract_pitch_statistics,
    train_fastpitch,
    synthesize_audio
)
import librosa

def main():
    # Étape 1 : Téléchargement des fichiers audio
    print("Téléchargement des fichiers audio...")
    urls, filenames, texts = download_audio_files(from_cache=True)
    
    # Étape 2 : Rééchantillonnage des fichiers audio
    print("Rééchantillonnage des fichiers audio...")
    resample_audio_files()
    
    # Étape 3 : Création et division du manifeste
    print("Création du manifeste...")
    create_manifest(urls, filenames, texts)
    split_manifest()

    # Étape 4 : Charger les modèles et le TextProcessor
    print("Chargement des modèles FastPitch et HiFi-GAN...")
    fastpitch, hifigan, denoiser, generator_train_setup, vocoder_train_setup = load_models()
    print("Chargement du TextProcessor...")
    tp = load_text_processing()
    
    # Étape 5 : Extraire les statistiques de pitch
    print("Extraction des statistiques de pitch...")
    train_manifest = "./audio/manifest_train.json"
    val_manifest = "./audio/manifest_validation.json"
    pitch_mean, pitch_std, pitch_min, pitch_max = extract_pitch_statistics(train_manifest)
    
    # Étape 6 : Entraîner le modèle FastPitch
    print("Entraînement du modèle FastPitch...")
    train_fastpitch(
        fastpitch,
        hifigan,
        denoiser,
        tp,
        train_manifest,
        val_manifest,
        pitch_mean,
        pitch_std,
        num_epochs=10,
        lr=1e-4,
        save_path="output/glados_fastpitch_model.pth"
    )

    # Étape 7 : Synthèse audio pour tester le modèle
    print("Synthèse audio avec FastPitch et HiFi-GAN...")
    sample_text = "Hello, this is a test of the FastPitch TTS model."
    generated_audio = synthesize_audio(
        fastpitch, hifigan, denoiser, sample_text, tp, pitch_mean, pitch_std
    )
    
    # Sauvegarder l'audio généré
    output_path = "./output/test_audio.wav"
    librosa.output.write_wav(output_path, generated_audio, sr=22050)
    print(f"Audio généré sauvegardé dans {output_path}")

if __name__ == "__main__":
    main()
