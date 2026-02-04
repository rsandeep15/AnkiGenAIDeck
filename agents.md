# Agents Playbook

This repo already wraps OpenAI + AnkiConnect; this page gives a fast “agent-first” map for Codex (or any LLM ops) to work safely and quickly.

## Core Surfaces
- **Scripts**: `AnkiSync.py` (PDF ➜ deck), `AnkiDeckToSpeech.py` (TTS audio), `AnkiDeckToImages.py` (image generation), Flask UI `app.py` (tabs for sync/audio/images/gallery/deck browser).
- **Backend APIs**:
  - OpenAI (text/audio/image models) via `openai` SDK.
  - AnkiConnect at `http://127.0.0.1:8765` using JSON RPC (see cheatsheet below).
- **Helpers**: `utils/common.py` for paths (`BASE_DIR`, `IMAGE_DIR`, `MEDIA_DIR`) and HTML/sound stripping regexes.

## Env & Secrets
- `.env` (used by Flask and scripts): `OPENAI_API_KEY`, `FLASK_APP=app.py`.
- Requires Anki running with AnkiConnect enabled.

## File/Dir Map
- Uploads: `uploads/`
- Media: `media/audio`, `media/images`
- Archived PDFs: `pdfs/`
- Flask static/templates: `static/`, `templates/`

## Concurrency & Rate Limits
- Audio workers: `--workers` or `ANKI_AUDIO_WORKERS` (default 10).
- Image workers: `--workers` or `ANKI_IMAGE_WORKERS` (default 3).
- Tuning these is the main lever to respect OpenAI rate limits.

## OpenAI Usage
- **Text extraction**: `AnkiSync.py` calls `client.responses.create` with `model` (e.g., `gpt-4.1-mini`, `gpt-4o-mini`) and an `input_file` (uploaded PDF). Prompt asks for JSON array of vocab pairs.
- **TTS**: `AnkiDeckToSpeech.py` uses `audio`/`tts` models (e.g., `gpt-4o-mini-tts`) and optional `voice`, `instructions`.
- **Images**: `AnkiDeckToImages.py` uses `image_model` (e.g., `gpt-image-1`) and optional gating prompt.
- **Model discovery**: `app.py` exposes `/api/models/<kind>` which filters the OpenAI model list into `text`, `audio`, `image`.

## AnkiConnect Cheatsheet
All requests: `{"action": "...", "params": {...}, "version": 6}` to `http://127.0.0.1:8765`.

Common calls:
- List decks: `deckNames`
- Find notes/cards: `findNotes`, `findCards` with queries like `deck:"MyDeck"`
- Note/card info: `notesInfo`, `cardsInfo`
- Create deck: `createDeck` with `deck`
- Change deck: `changeDeck` with `cards` (card IDs) and `deck`
- Add media: `storeMediaFile` with `filename`, `data` (base64)
- Update note fields: `updateNoteFields` with `note` (note ID) and `fields`
- Delete deck (careful): `deleteDecks` requires `cardsToo=true` even if empty

## Flask API Surface
- `/sync` (POST FormData): `file` (PDF), `deck`, `model`, `romanized`
- `/generate/audio` (POST JSON): `deck`, `model`, `voice?`, `instructions?`, `workers?`
- `/generate/images` (POST JSON): `deck`, `image_model`, `prompt?`, `skip_gating?`, `workers?`
- `/api/decks`, `/api/models/<kind>`, `/api/deck-images`, `/api/deck-cards`
- Static UI: `templates/index.html`, logic in `static/app.js`

## Safe Ops Notes
- AnkiConnect deletion: never call `deleteDecks` unless decks are empty or you set `cardsToo=true` deliberately.
- Sound/HTML stripping: use `clean_field_text` logic in `app.py` (or `utils/common.py` regexes) to avoid leaking `[sound:...]` into displays.
- Network: OpenAI calls require `OPENAI_API_KEY`; local AnkiConnect only.

## Task Recipes (for agents)
- **Sync a PDF**: `python AnkiSync.py path/to.pdf --deck "MyDeck" --model gpt-4.1-mini --romanized/--no-romanized`
- **Generate audio**: `python AnkiDeckToSpeech.py "MyDeck" --model gpt-4o-mini-tts --voice onyx --workers 8`
- **Generate images**: `python AnkiDeckToImages.py "MyDeck" --image-model gpt-image-1 --workers 3 --skip-gating`
- **Run UI**: `flask run` (with venv + `.env`), open http://127.0.0.1:5000/
- **List decks (AnkiConnect)**: POST `{"action":"deckNames","params":{},"version":6}` to `http://127.0.0.1:8765`

## Converting to “agents-first”
- Keep this doc updated with: latest OpenAI models in use, worker defaults, and any prompt/gating tweaks.
- Add new task recipes as flows change (e.g., new tabs/endpoints).
- When adding automation, reuse existing AnkiConnect helpers (`invoke` pattern) and respect deck change safety above.
