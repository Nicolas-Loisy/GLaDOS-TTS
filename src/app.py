import argparse
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
    parser = argparse.ArgumentParser(description="GLaDOS-TTS Training and Generation")
    parser.add_argument('--train', action='store_true', help="Activer l'entraînement du modèle")
    parser.add_argument('--generate', action='store_true', help="Activer la génération audio")
    parser.add_argument('--text', type=str, help="Texte à synthétiser (pour la génération)")
    parser.add_argument('--model_path', type=str, help="Chemin vers le modèle FastPitch à charger (pour la génération)")
    args = parser.parse_args()

    if args.train:
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

    if args.generate:
        if not args.text:
            raise ValueError("Le texte à synthétiser doit être fourni avec l'option --text")
        
        # Charger les modèles et le TextProcessor
        fastpitch, hifigan, denoiser, generator_train_setup, vocoder_train_setup = load_models()
        tp = load_text_processing()

        # Synthétiser l'audio
        audio = synthesize_audio(
            fastpitch, hifigan, denoiser, args.text, tp,
            pitch_mean=0, pitch_std=1, model_path=args.model_path
        )

        # Sauvegarder l'audio généré
        output_path = "output/generated_audio.wav"
        librosa.output.write_wav(output_path, audio, sr=22050)
        print(f"Audio généré sauvegardé dans {output_path}")

if __name__ == "__main__":
    main()
