import os
import shutil
import soundfile as sf
from tqdm import tqdm
from config import TEMP_DIR, AUDIO_DIR, SAMPLING_RATE

def audio_duration(file_path):
    """Calculate the duration of an audio file in seconds."""
    with sf.SoundFile(file_path) as f:
        return f.frames / f.samplerate

def resample_audio(input_path, output_path, target_sampling_rate):
    """Resample an audio file to the target sampling rate."""
    import librosa
    audio, _ = librosa.load(input_path, sr=target_sampling_rate)
    sf.write(output_path, audio, samplerate=target_sampling_rate)

def resample_audio_files():
    print("Resampling audio files...")
    os.makedirs(TEMP_DIR, exist_ok=True)

    for filename in tqdm(os.listdir(AUDIO_DIR)):
        if filename.endswith(".wav"):
            source_path = os.path.join(AUDIO_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            resample_audio(source_path, temp_path, SAMPLING_RATE)

    shutil.rmtree(AUDIO_DIR)
    shutil.move(TEMP_DIR, AUDIO_DIR)
