# Agents Playbook

This repo already wraps OpenAI + AnkiConnect; this page gives a fast “agent-first” map for Codex (or any LLM ops) to work safely and quickly.

## Tool-First Rule (Important)
- Prefer the canonical wrapper tools before writing bespoke AnkiConnect code.
- Start discovery with `agent_tools.json` and `scripts/agent_tools.py`.
- Only drop to raw AnkiConnect calls when the wrapper cannot perform the requested task.
- If using raw AnkiConnect, explain why wrapper tools were insufficient.

## Core Surfaces
- **Scripts**: `AnkiSync.py` (PDF ➜ deck), `AnkiDeckToSpeech.py` (TTS audio), `AnkiDeckToImages.py` (image generation), Flask UI `app.py` (tabs for sync/audio/images/gallery/deck browser).
- **Agent wrapper**: `scripts/agent_tools.py` with machine-readable manifest in `agent_tools.json`.
- **Backend APIs**:
  - OpenAI (text/audio/image models) via `openai` SDK.
  - AnkiConnect at `http://127.0.0.1:8765` using JSON RPC (see cheatsheet below).
- **Helpers**: `utils/common.py` for paths (`BASE_DIR`, `IMAGE_DIR`, `MEDIA_DIR`) and HTML/sound stripping regexes.

## Agent Tool Registry
- List tools (machine-readable): `.venv/bin/python scripts/agent_tools.py list`
- Manifest location: `agent_tools.json`
- Wrapper entrypoint: `scripts/agent_tools.py`

## Env & Secrets
- `.env` (used by Flask and scripts): `OPENAI_API_KEY`, `FLASK_APP=app.py`.
- Requires Anki running with AnkiConnect enabled.

## File/Dir Map
- Uploads: `uploads/`
- Media: `media/audio`, `media/images`
- Archived PDFs: `.local_memory/pdfs/`
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
- **Sync a PDF**: `.venv/bin/python scripts/agent_tools.py sync --pdf path/to.pdf --deck "MyDeck" --model gpt-4.1-mini --romanized`
- **Generate audio**: `.venv/bin/python scripts/agent_tools.py audio --deck "MyDeck" --model gpt-4o-mini-tts --voice onyx --workers 10`
- **Generate images**: `.venv/bin/python scripts/agent_tools.py images --deck "MyDeck" --image-model gpt-image-1 --workers 3`
- **Import manual card pairs**: `.venv/bin/python scripts/agent_tools.py cards-import --deck "MyDeck" --input pairs.json --front-key Front --back-key Back --dry-run`
- **Run UI**: `.venv/bin/python scripts/agent_tools.py ui --port 5000`, then open http://127.0.0.1:5000/
- **List decks (AnkiConnect)**: POST `{"action":"deckNames","params":{},"version":6}` to `http://127.0.0.1:8765`

## Converting to “agents-first”
- Keep this doc updated with: latest OpenAI models in use, worker defaults, and any prompt/gating tweaks.
- Add new task recipes as flows change (e.g., new tabs/endpoints).
- When adding automation, reuse existing AnkiConnect helpers (`invoke` pattern) and respect deck change safety above.
