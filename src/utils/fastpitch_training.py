import os
import json
import librosa
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# 1. Charger le modèle FastPitch
def load_fastpitch_model():
    """Charge le modèle FastPitch à partir de NVIDIA/DeepLearningExamples."""
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch', trust_repo=True)
    model.eval()  # Mode évaluation par défaut
    return model

# 2. Extraire les statistiques de pitch
def extract_pitch_statistics(manifest_path):
    """Extrait les statistiques de pitch (mean, std, min, max) à partir des fichiers audio listés dans le manifeste."""
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
        
        pitch_values.extend(pitches)
    
    # Calculer les statistiques
    pitch_mean = np.mean(pitch_values)
    pitch_std = np.std(pitch_values)
    pitch_min = np.min(pitch_values)
    pitch_max = np.max(pitch_values)
    
    return pitch_mean, pitch_std, pitch_min, pitch_max

# 3. Dataset pour FastPitch
class FastPitchDataset(Dataset):
    def __init__(self, manifest_path):
        # Charger le manifeste JSON comme un tableau
        with open(manifest_path, "r") as f:
            self.data = json.load(f)  # Charger tout le tableau JSON

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        audio, sr = librosa.load(entry["audio_filepath"], sr=None)
        text = entry["text"]
        return audio, text

# 4. Entraîner le modèle FastPitch
def train_fastpitch(model, train_manifest, val_manifest, pitch_mean, pitch_std, pitch_min, pitch_max, num_epochs=10, lr=1e-4):
    """Entraîne le modèle FastPitch."""
    # Préparer les données
    train_dataset = FastPitchDataset(train_manifest)
    val_dataset = FastPitchDataset(val_manifest)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=6)
    
    # Configurer l'optimiseur
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Entraîner le modèle
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            audio, text = batch
            # Prétraitement et passage dans le modèle
            # Assurez-vous d'adapter cela à l'architecture de FastPitch
            outputs = model(audio, text, pitch_mean=pitch_mean, pitch_std=pitch_std)
            
            # Calcul de la perte (à définir selon le modèle)
            loss = outputs.loss  # Exemple, dépend du modèle exact
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                audio, text = batch
                outputs = model(audio, text, pitch_mean=pitch_mean, pitch_std=pitch_std)
                val_loss += outputs.loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")
