import os
import shutil
from tqdm.notebook import tqdm
from utils.utils import resample_audio
from config import TEMP_DIR, AUDIO_DIR, SAMPLING_RATE

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
