# Gen AI Anki Toolkit

Streamline your language decks with a trio of OpenAI-powered helpers:

- `AnkiSync.py` turns vocab PDFs into fully-populated Anki decks.
- `AnkiDeckToSpeech.py` adds natural-sounding audio pronunciations.
- `AnkiDeckToImages.py` decorates cards with visual mnemonics.
- `app.py` (optional) launches a local Flask UI for drag-and-drop syncing, media generation, and deck tutoring chat.

All scripts talk to a local AnkiConnect instance at `http://127.0.0.1:8765` and assume `OPENAI_API_KEY` is set in your shell.

---

## Agent-First Wrapper CLI

Use `scripts/agent_tools.py` as the canonical entrypoint for agents (Codex/Claude) before writing bespoke AnkiConnect calls.

```bash
.venv/bin/python scripts/agent_tools.py list
```

Supported workflows:

- `sync`: PDF -> deck
- `audio`: deck audio generation
- `images`: deck image generation
- `cards-import`: import explicit Front/Back pairs from JSON
- `docs-list`: list local memory PDFs
- `docs-read`: read a specific page from a local memory PDF
- `ui`: run Flask app

Examples:

```bash
# PDF -> deck
.venv/bin/python scripts/agent_tools.py sync \
  --pdf /path/to/lesson.pdf \
  --deck "AdvancedBeginner1::LessonX" \
  --model gpt-4.1-mini \
  --romanized

# Audio
.venv/bin/python scripts/agent_tools.py audio \
  --deck "AdvancedBeginner1" \
  --model gpt-4o-mini-tts \
  --voice onyx \
  --workers 10

# Images
.venv/bin/python scripts/agent_tools.py images \
  --deck "AdvancedBeginner1::Emotions & States" \
  --image-model gpt-image-1 \
  --workers 3

# Manual card import
.venv/bin/python scripts/agent_tools.py cards-import \
  --deck "AdvancedBeginner1::Lesson3::GeotGatayo_13_Phrases" \
  --input /path/to/pairs.json \
  --front-key Front \
  --back-key Back \
  --dry-run

# Local memory docs (agent-readable)
.venv/bin/python scripts/agent_tools.py docs-list
.venv/bin/python scripts/agent_tools.py docs-read --file Toy_Korean_Verbs_Table.pdf --page 1
```

`cards-import` accepts either:

- a JSON array of objects, e.g. `[{"Front":"...","Back":"..."}]`
- or a wrapped object, e.g. `{"cards":[{"front":"...","back":"..."}]}`

Tool metadata is in `agent_tools.json`.

### Local memory docs

- Put private/reference PDFs in `.local_memory/pdfs/` for agent lookup via `docs-list` and `docs-read`.
- `.local_memory/` is ignored by git by default.
- A single shareable sample is committed for demos:
  - `.local_memory/pdfs/Toy_Korean_Verbs_Table.pdf`

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# store secrets in .env so both CLI and Flask server can read them
cat <<'EOF' > .env
OPENAI_API_KEY=YOUR_KEY_HERE
FLASK_APP=app.py
EOF

# (optional) export for current shell if you plan to run scripts directly
export OPENAI_API_KEY=YOUR_KEY_HERE
export FLASK_APP=app.py
```

Make sure Anki is running with the AnkiConnect add-on enabled. Generated media files are stored under `media/audio`, `media/images`, and processed PDFs are archived to `pdfs/`.

### Pre-commit (optional)

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Pre-commit will run basic sanity checks on staged files, plus `python -m compileall` and the unit tests in `tests/`.

To launch the web UI, run `flask run` (or `python app.py`) after activating the virtualenv. The server will load credentials from `.env`.

### Web UI quickstart

```bash
source .venv/bin/activate
flask run  # or python app.py
```

Then open http://127.0.0.1:5000/ in your browser. Drag-and-drop a PDF, tweak the deck/model options, and click “Upload & Sync” to trigger `AnkiSync.py`.

Switch to the **Deck Audio** or **Deck Images** tabs to:

- pick an existing deck from a live AnkiConnect dropdown
- trigger `AnkiDeckToSpeech.py` or `AnkiDeckToImages.py` without the CLI
- monitor stdout/stderr for each job directly in the browser
- watch optimistic progress/ETA updates while long-running jobs finish
- choose from your account’s available OpenAI models via the auto-populated dropdowns
- control concurrency with worker dropdowns that mirror the script defaults

There is also a **Deck Chat** tab that lets you pick a deck, choose a response model, and ask tutoring questions.
Answers stream live in a chat-style view (Enter to send, Shift+Enter for newline), with multi-turn context in-session.

The **Deck Images** tab now includes the image gallery with pagination and per-deck image coverage.
The **Deck Browser** tab shows word pairs in a scrollable table.

---

## AnkiSync — Build Decks From PDFs

**What it does**

- uploads a PDF to OpenAI
- extracts vocabulary pairs (`english`, `foreign`, optional `romanized`)
- creates / populates the target deck via AnkiConnect
- archives the original PDF to `pdfs/`

**Usage**

```bash
python AnkiSync.py path/to/lesson.pdf --deck "Korean Deck" \
  --model gpt-4.1-mini \
  --romanized          # include romanized text (default)
```

Key flags:

- `--deck`: overrides the auto-generated deck name (defaults to the PDF filename without extension)
- `--model`: choose the extraction model (e.g. `gpt-4o-mini`, `gpt-4.1`)
- `--romanized` / `--no-romanized`: toggle romanized text in card fronts

Failures emit the offending JSON snippet to help diagnose prompt/output issues.

---

## AnkiDeckToSpeech — Add Pronunciation Audio

**What it does**

- fetches notes from the chosen deck
- skips cards that already include `[sound ...]`
- generates MP3s in `media/audio/` using OpenAI TTS
- attaches the audio to the front field via AnkiConnect

**Usage**

```bash
python AnkiDeckToSpeech.py "Korean Deck" \
  --model gpt-4o-mini-tts \
  --voice onyx \
  --workers 8
```

Key flags:

- `--model`: any supported TTS model (`gpt-4o-mini-tts`, etc.)
- `--voice`: voice preset offered by the TTS model
- `--instructions`: extra voice guidance (defaults to “speak like a native ... ignore HTML/parentheses”)
- `--workers`: concurrency level (defaults to `ANKI_AUDIO_WORKERS` env var or 10)

Text is sanitized before synthesis (HTML stripped, whitespace collapsed). The script finishes with a summary of added / skipped / failed generations.

---

## AnkiDeckToImages — Add Visual Mnemonics

**What it does**

- retrieves deck notes and replaces any existing `<img>` tags with regenerated art
- (optional) runs a prompt-configured gating check to decide whether an image is helpful
- generates PNGs in `media/images/` using the configured image model
- attaches the image to the front field via AnkiConnect

**Usage**

```bash
python AnkiImageGen.py "Korean Deck" \
  --image-model gpt-image-1 \
  --prompt "Generate a mnemonic illustration for: {text}" \
  --workers 3
```

Key flags:

- `--image-model`: OpenAI image endpoint to call
- `--prompt`: templated string where `{text}` is replaced with the card back
- `--workers`: concurrency level (defaults to `ANKI_IMAGE_WORKERS` env var or 3)
- `--skip-gating`: generate for every card, bypassing the saved gating prompt

Each run ends with a summary of added / skipped / failed image generations.
Images are attached to the **Back** field of the card.

---

## Tips & Troubleshooting

- **Rate limits**: tune `--workers` (or env vars) to stay within your OpenAI quotas.
- **AnkiConnect errors**: ensure Anki is open, add-on installed, and port accessible.
- **Logging verbosity**: scripts print card-level status messages; redirect stdout if you prefer a quieter run.
- **Web UI**: the Flask server uploads PDFs to `./uploads/` before invoking `AnkiSync.py`.

Happy deck building! Feel free to mix and match scripts—import the vocab with `AnkiSync.py`, then layer on audio and images whenever you’re ready.
