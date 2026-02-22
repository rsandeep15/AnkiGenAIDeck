from pathlib import Path

# Core paths
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
AUDIO_DIR = MEDIA_DIR / "audio"
IMAGE_DIR = MEDIA_DIR / "images"

# AnkiConnect
ANKI_CONNECT_URL = "http://127.0.0.1:8765"

# Defaults
DEFAULT_TEXT_MODEL = "gpt-4.1-mini"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_IMAGE_MODEL = "gpt-image-1"
DEFAULT_TTS_WORKERS = 10
DEFAULT_IMAGE_WORKERS = 3

# OpenAI storage (for evals/logging). When False, do not request server-side storage.
OPENAI_STORE_RESPONSES = True

# Gating prompt for images
GATING_PROMPT_ID = "pmpt_69194beaad7c819497842682bad97629040fc2c239b73233"
GATING_PROMPT_VERSION = "5"
