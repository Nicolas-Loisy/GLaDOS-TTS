import os
from dotenv import load_dotenv


AUDIO_DIR = 'audio'
TEMP_DIR = 'temp_audio'
CACHE_FILE = 'cache/cache.json'
DOWNLOAD_THREADS = 64
SAMPLING_RATE = 22050
BASE_DIR = "glados_out"

blocklist = ["potato", "_ding_", "00_part1_entry-6", "_escape_"]
sources = ["https://theportalwiki.com/wiki/GLaDOS_voice_lines/fr"]


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            cls._load_env()
        return cls._instance

    @classmethod
    def _load_env(cls):
        load_dotenv()

    def get_env_var(self, var_name, default_value=None):
        value = os.getenv(var_name)
        if value is None:
            if default_value is not None:
                value = default_value
            else:
                raise EnvironmentError(f"Environment variable {var_name} not found")
        return value

    def get_env_var_list(self, var_name, default_value=None):
        value = self.get_env_var(var_name, default_value)
        return set(value.split(","))
