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
        prepared = self.tp.prepare_input_sequence([text], batch_size=1)
        text_tensor = prepared[0]['text']  # Assurez-vous que c'est correct
        return audio, text_tensor.squeeze()

# Fonction de collate pour gérer les tailles dynamiques
def collate_fn(batch):
    """
    Fonction de collate pour ajouter du padding aux données audio et textes.
    Cette fonction gère également les séquences vides et les types de données.
    """
    
    # Filtrer les éléments vides dans le batch (audios et textes)
    batch = [item for item in batch if (item[0].ndim > 0 and item[1].ndim > 0) and len(item[0]) > 0 and len(item[1]) > 0]
    
    # Si après le filtrage, le batch est vide, lever une erreur
    if len(batch) == 0:
        raise ValueError("Le batch contient uniquement des séquences vides.")
    
    # Extraire les audios et les textes
    audios = [torch.tensor(item[0]) for item in batch]  # Convertir les audios en tensors
    texts = [item[1] for item in batch]  # Les textes sont déjà des tensors
    
    # Ajouter du padding pour les audios
    padded_audios = pad_sequence(audios, batch_first=True)
    
    # Ajouter du padding pour les textes, s'assurer que chaque texte est un tensor
    texts = [torch.tensor(t).squeeze() if not isinstance(t, torch.Tensor) else t.squeeze() for t in texts]
    padded_texts = pad_sequence(texts, batch_first=True)
    
    return padded_audios, padded_texts

def pitch_transform(pitch_pred, mask):
    """
    Fonction de transformation du pitch avec un masque.
    """
    # Vérification des dimensions
    if pitch_pred.size(1) != mask.size(1):
        raise ValueError(f"Les dimensions de pitch_pred ({pitch_pred.size(1)}) et mask ({mask.size(1)}) ne correspondent pas.")
    
    # Étendre le masque pour correspondre à la taille de pitch_pred
    mask = mask.unsqueeze(-1)  # Ajouter une dimension pour correspondre à pitch_pred
    
    # Appliquer la transformation du pitch
    return pitch_pred * mask


# 4. Entraîner le modèle FastPitch
def train_fastpitch(
    fastpitch, hifigan, denoiser, tp, train_manifest, val_manifest,
    pitch_mean, pitch_std, num_epochs=10, lr=1e-4, save_path="output/glados_fastpitch_model.pth"
):
    """Entraîne le modèle FastPitch."""
    # Préparer les données
    train_dataset = FastPitchDataset(train_manifest, tp)
    val_dataset = FastPitchDataset(val_manifest, tp)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=6, collate_fn=collate_fn)
    
    # Configurer l'optimiseur
    optimizer = Adam(fastpitch.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        fastpitch.train()
        for batch in train_loader:
            audio, text = batch

            # Si text est une liste, appliquez le padding
            if isinstance(text, list):
                text = [torch.tensor(t).squeeze() if not isinstance(t, torch.Tensor) else t.squeeze() for t in text]
                text = pad_sequence(text, batch_first=True)

            # Log des dimensions après le padding
            print(f"Dimensions de text après padding : {text.size()}")

            # Génération des paramètres pour FastPitch
            gen_kw = {
                'pace': 1.0,
                'speaker': 0,
                'pitch_tgt': None,
                'pitch_transform': None
                # 'pitch_tgt': torch.full(text.size(), pitch_mean, dtype=text.dtype, device=text.device),
                # 'pitch_transform': pitch_transform
            }

            # Passe dans le modèle FastPitch
            outputs = fastpitch(text, **gen_kw)

            # Calcul de la perte
            # loss = outputs.loss  # Exemple, dépend de votre implémentation
            # print(f"Perte : {loss.item()}")

            # # Backpropagation
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        # Validation
        fastpitch.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                audio, text = batch
                if isinstance(text, list):
                    text = [torch.tensor(t).squeeze() if not isinstance(t, torch.Tensor) else t.squeeze() for t in text]
                    text = pad_sequence(text, batch_first=True)

                gen_kw = {
                    'pace': 1.0,
                    'speaker': 0,
                    'pitch_tgt': None,
                    'pitch_transform': None
                    # 'pitch_tgt': torch.full(text.size(), pitch_mean, dtype=text.dtype, device=text.device),
                    # 'pitch_transform': lambda x, y: x * pitch_std  # Même transformation
                }
                outputs = fastpitch(text, **gen_kw)
                # val_loss += outputs.loss.item()
            # print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")

        # Sauvegarde du modèle après chaque époque
        epoch_save_path = save_path.replace(".pth", f"_epoch{epoch + 1}.pth")
        torch.save(fastpitch.state_dict(), epoch_save_path)
        print(f"Modèle sauvegardé : {epoch_save_path}")

# 5. Synthèse audio avec HiFi-GAN
def synthesize_audio(fastpitch, hifigan, denoiser, text, tp, pitch_mean, pitch_std, model_path=None):
    """
    Génère un fichier audio à partir d'un texte.
    
    Args:
        fastpitch: Modèle FastPitch.
        hifigan: Modèle HiFi-GAN.
        denoiser: Denoiser pour HiFi-GAN.
        text: Texte à synthétiser.
        tp: Préprocesseur de texte.
        pitch_mean: Moyenne du pitch pour la normalisation.
        pitch_std: Écart-type du pitch pour la normalisation.
        model_path: Chemin vers le modèle FastPitch à charger (optionnel).
    """
    # Charger le modèle si un chemin est spécifié
    if model_path:
        fastpitch.load_state_dict(torch.load(model_path))
        fastpitch.eval()  # Mettre le modèle en mode évaluation
        print(f"Modèle chargé depuis : {model_path}")
    
    # Préparer la séquence d'entrée
    batch = tp.prepare_input_sequence([text], batch_size=1)

    # Vérifier le contenu de batch
    print(f"Contenu de batch : {batch}")
    
    # Extraire les indices de texte depuis le batch
    if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict) and 'text' in batch[0]:
        text_input = batch[0]['text']  # Tensor contenant les indices de texte
    else:
        raise KeyError(f"Impossible de trouver les indices nécessaires dans batch : {batch}")
    
    # Générer les paramètres pour le modèle FastPitch
    gen_kw = {'pace': 1.0, 'speaker': 0, 'pitch_tgt': pitch_mean, 'pitch_transform': pitch_std}
    
    with torch.no_grad():
        # Passer par le modèle FastPitch
        mel, _, *_ = fastpitch(text_input, **gen_kw)
        
        # Générer l'audio avec HiFi-GAN
        audio = hifigan(mel).float()

        denoising_strength=0.005
        
        # Appliquer le denoiser
        audio = denoiser(audio.squeeze(1), denoising_strength)
        
        # Finaliser l'audio
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
    
    print(f"Manifeste d'entraînement sauvegardé dans {train_manifest}")
    print(f"Manifeste de validation sauvegardé dans {val_manifest}")
