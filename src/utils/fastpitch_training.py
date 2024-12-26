import os
import json
import librosa
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 1. Charger les modèles FastPitch et HiFi-GAN
def load_models():
    """Charge les modèles FastPitch et HiFi-GAN."""
    fastpitch, generator_train_setup = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch', trust_repo=True
    )
    hifigan, vocoder_train_setup, denoiser = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan', trust_repo=True
    )
    
    # Passer les modèles en mode évaluation
    fastpitch.eval()
    hifigan.eval()
    denoiser.eval()
    
    return fastpitch, hifigan, denoiser, generator_train_setup, vocoder_train_setup

# 2. Prétraitement du texte avec NVIDIA TextProcessing
def load_text_processing():
    """Charge le préprocesseur de texte NVIDIA."""
    tp = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_textprocessing_utils',
        cmudict_path="data/cmudict/cmudict-0.7b",
        heteronyms_path="heteronyms"
    )
    return tp

# 3. Dataset pour FastPitch
class FastPitchDataset(Dataset):
    def __init__(self, manifest_path, tp):
        # Charger le manifeste JSON comme un tableau
        with open(manifest_path, "r") as f:
            self.data = json.load(f)
        self.tp = tp  # TextProcessor pour préparer les entrées

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        audio, sr = librosa.load(entry["audio_filepath"], sr=None)
        text = entry["text"]
        
        # Préparer le texte avec le TextProcessor
        text_tensor = self.tp.prepare_input_sequence([text], batch_size=1)[0]['text']
        return audio, text_tensor

# Fonction de collate pour gérer les tailles dynamiques
def collate_fn(batch):
    """
    Fonction de collate pour ajouter du padding aux données audio et textes.
    """
    audios = [torch.tensor(item[0]) for item in batch]  # Convertir les audios en tensors
    texts = [item[1] for item in batch]  # Les textes sont déjà des tensors
    
    # Ajouter du padding pour les audios
    padded_audios = pad_sequence(audios, batch_first=True)
    return padded_audios, texts

# 4. Entraîner le modèle FastPitch
def train_fastpitch(
    fastpitch, hifigan, denoiser, tp, train_manifest, val_manifest,
    pitch_mean, pitch_std, num_epochs=10, lr=1e-4
):
    """Entraîne le modèle FastPitch."""
    # Préparer les données
    train_dataset = FastPitchDataset(train_manifest, tp)
    val_dataset = FastPitchDataset(val_manifest, tp)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=6, collate_fn=collate_fn)
    
    # Configurer l'optimiseur
    optimizer = Adam(fastpitch.parameters(), lr=lr)
    
    # Entraîner le modèle
    for epoch in range(int(num_epochs)):
        fastpitch.train()
        for batch in train_loader:
            audio, text = batch
            gen_kw = {
                'pace': 1.0,
                'speaker': 0,
                'pitch_tgt': torch.full_like(text, pitch_mean),  # Moyenne du pitch comme cible
                'pitch_transform': pitch_std  # Écart type du pitch pour la transformation
            }
            outputs = fastpitch(text, **gen_kw)  # Passe le texte avec les hyperparamètres
            
            # Calcul de la perte (exemple, dépend du modèle exact)
            loss = outputs.loss  # Peut varier selon votre implémentation
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        fastpitch.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                audio, text = batch
                gen_kw = {
                    'pace': 1.0,
                    'speaker': 0,
                    'pitch_tgt': torch.full_like(text, pitch_mean),
                    'pitch_transform': pitch_std
                }
                outputs = fastpitch(text, **gen_kw)
                val_loss += outputs.loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")

# 5. Synthèse audio avec HiFi-GAN
def synthesize_audio(fastpitch, hifigan, denoiser, text, tp, pitch_mean, pitch_std):
    """Génère un fichier audio à partir d'un texte."""
    batch = tp.prepare_input_sequence([text], batch_size=1)
    gen_kw = {'pace': 1.0, 'speaker': 0, 'pitch_tgt': pitch_mean, 'pitch_transform': pitch_std}
    
    with torch.no_grad():
        mel, _, *_ = fastpitch(batch['text'], **gen_kw)
        audio = hifigan(mel).float()
        audio = denoiser(audio.squeeze(1), denoising_strength=0.005)
        audio = audio.squeeze(1).detach().cpu().numpy()
    return audio

# 6. Fonction pour extraire les statistiques de pitch
def extract_pitch_statistics(manifest_path):
    """
    Extrait les statistiques de pitch (mean, std, min, max) à partir des fichiers audio listés dans le manifeste.
    """
    pitch_values = []
    
    # Charger le manifeste JSON comme un tableau
    with open(manifest_path, "r") as f:
        data = json.load(f)  # Charger tout le tableau JSON
    
    for entry in data:
        audio_path = entry["audio_filepath"]
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculer le pitch (F0) avec librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[magnitudes > np.median(magnitudes)]  # Filtrer les valeurs faibles
        
        if len(pitches) > 0:
            pitch_values.extend(pitches)
    
    # Calculer les statistiques
    pitch_mean = np.mean(pitch_values) if pitch_values else 0
    pitch_std = np.std(pitch_values) if pitch_values else 1e-6  # Éviter la division par zéro
    pitch_min = np.min(pitch_values) if pitch_values else 0
    pitch_max = np.max(pitch_values) if pitch_values else 0
    
    return pitch_mean, pitch_std, pitch_min, pitch_max

# 7. Fonction pour créer le manifeste
def create_manifest(urls, filenames, texts, output_path="./audio/manifest.json"):
    """
    Crée un fichier manifeste JSON pour les fichiers audio.
    Chaque entrée contient le chemin du fichier audio, le texte et la durée.
    """
    manifest = []
    for url, filename, text in zip(urls, filenames, texts):
        audio_path = os.path.join("./audio", filename)
        if os.path.exists(audio_path):
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            manifest.append({
                "audio_filepath": audio_path,
                "text": text,
                "duration": duration
            })
    
    # Sauvegarder le manifeste dans un fichier
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifeste sauvegardé dans {output_path}")

# 8. Fonction pour diviser le manifeste
def split_manifest(input_path="./audio/manifest.json", train_ratio=0.8):
    """
    Divise le manifeste en fichiers d'entraînement et de validation.
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Sauvegarder les fichiers divisés
    train_manifest = "./audio/manifest_train.json"
    val_manifest = "./audio/manifest_validation.json"
    
    with open(train_manifest, "w") as f:
        json.dump(train_data, f, indent=4)
    
    with open(val_manifest, "w") as f:
        json.dump(val_data, f, indent=4)
    
    print(f"Manifeste divisé : {train_manifest} et {val_manifest}")

# 9. Fonction pour télécharger et rééchantillonner les fichiers audio
def resample_audio_files(input_dir="./audio", target_sr=22050):
    """
    Rééchantillonne tous les fichiers audio dans le répertoire donné à la fréquence cible.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            y, sr = librosa.load(input_path, sr=None)
            if sr != target_sr:
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                librosa.output.write_wav(input_path, y_resampled, sr=target_sr)
                print(f"Rééchantillonné : {filename}")

# 10. Fonction pour télécharger les fichiers audio
def download_audio_files(from_cache=True, cache_path="./audio_cache.json"):
    """
    Télécharge ou charge les fichiers audio à partir du cache.
    """
    if from_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        urls, filenames, texts = data["urls"], data["filenames"], data["texts"]
        print("Chargé depuis le cache.")
        return urls, filenames, texts
    
    # Exemple de téléchargement (remplacer par votre propre logique)
    urls = ["https://example.com/audio1.wav", "https://example.com/audio2.wav"]
    filenames = ["audio1.wav", "audio2.wav"]
    texts = ["This is the first audio.", "This is the second audio."]
    
    # Simuler le téléchargement
    for url, filename in zip(urls, filenames):
        # Télécharger ici
        print(f"Téléchargement simulé : {url} -> {filename}")
    
    # Sauvegarder dans le cache
    with open(cache_path, "w") as f:
        json.dump({"urls": urls, "filenames": filenames, "texts": texts}, f)
    
    return urls, filenames, texts
