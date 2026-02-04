import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from openai import OpenAI
from AnkiSync import invoke
from utils.common import (
    BASE_DIR,
    IMAGE_DIR,
    MEDIA_DIR,
    HTML_TAG_RE,
    IMG_SRC_RE,
    SOUND_TAG_RE,
    NBSP_RE,
)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def run_script(script_path: Path, args: List[str]):
    env = os.environ.copy()
    if "OPENAI_API_KEY" not in env:
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")

    command = [sys.executable, str(script_path)] + args
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/decks", methods=["GET"])
def list_decks():
    try:
        decks = invoke("deckNames")
        return jsonify({"ok": True, "decks": decks})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@lru_cache(maxsize=1)
def cached_model_ids() -> List[str]:
    client = OpenAI()
    response = client.models.list()
    data = getattr(response, "data", [])
    return [getattr(model, "id", "") for model in data if getattr(model, "id", "")]


def filter_models(kind: str) -> List[str]:
    kind = kind.lower()
    ids = cached_model_ids()

    def is_text(model_id: str) -> bool:
        blocked = ("tts", "audio", "image", "embed", "embedding", "speech")
        if not model_id.startswith(("gpt", "o")):
            return False
        return not any(token in model_id for token in blocked)

    def is_audio(model_id: str) -> bool:
        return "tts" in model_id or model_id.endswith("-tts") or "audio" in model_id

    def is_image(model_id: str) -> bool:
        return "image" in model_id or model_id.startswith("dall-e")

    if kind == "text":
        return sorted(filter(is_text, ids))
    if kind == "audio":
        return sorted(filter(is_audio, ids))
    if kind == "image":
        return sorted(filter(is_image, ids))
    raise ValueError("Unsupported model kind")


@app.route("/api/models/<kind>", methods=["GET"])
def list_models(kind: str):
    if kind not in {"text", "audio", "image"}:
        return jsonify({"ok": False, "message": "Unsupported model type."}), 400

    if request.args.get("refresh") == "1":
        cached_model_ids.cache_clear()

    try:
        models = filter_models(kind)
        if not models:
            return jsonify({"ok": False, "message": f"No {kind} models available."}), 404
        return jsonify({"ok": True, "models": models})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/api/deck-images", methods=["GET"])
def deck_images():
    deck = request.args.get("deck", "").strip()
    if not deck:
        return jsonify({"ok": False, "message": "Deck parameter is required."}), 400

    try:
        note_ids = invoke("findNotes", query=f'deck:"{deck}"')
        if not note_ids:
            return jsonify({"ok": True, "images": []})
        notes = invoke("notesInfo", notes=note_ids)
        results = []
        for note_id, note in zip(note_ids, notes):
            front = note["fields"]["Front"]["value"]
            back = note["fields"]["Back"]["value"]
            filename = extract_image_filename(front) or extract_image_filename(back)
            if not filename:
                continue
            local_path = IMAGE_DIR / filename
            if not local_path.exists():
                stem, suffix = os.path.splitext(filename)
                if "-" in stem:
                    base = stem.split("-", 1)[0] + suffix
                    alt_path = IMAGE_DIR / base
                    if alt_path.exists():
                        local_path = alt_path
                    else:
                        continue
                else:
                    continue
            results.append(
                {
                    "card_id": note_id,
                    "english": clean_field_text(back),
                    "front_text": clean_field_text(front),
                    "image_url": url_for("serve_image_file", filename=local_path.name),
                }
            )
        return jsonify({"ok": True, "images": results})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/api/deck-cards", methods=["GET"])
def deck_cards():
    deck = request.args.get("deck", "").strip()
    if not deck:
        return jsonify({"ok": False, "message": "Deck parameter is required."}), 400

    try:
        note_ids = invoke("findNotes", query=f'deck:"{deck}"')
        if not note_ids:
            return jsonify({"ok": True, "cards": []})
        notes = invoke("notesInfo", notes=note_ids)
        cards = []
        for note_id, note in zip(note_ids, notes):
            fields = note.get("fields", {})
            front = clean_field_text((fields.get("Front") or {}).get("value", ""))
            back = clean_field_text((fields.get("Back") or {}).get("value", ""))
            romanized = clean_field_text((fields.get("Romanized") or {}).get("value", ""))
            cards.append(
                {
                    "id": note_id,
                    "front": front,
                    "back": back,
                    "romanized": romanized,
                }
            )
        return jsonify({"ok": True, "cards": cards})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


def estimate_sync_duration(card_count: int) -> Tuple[int, str]:
    if card_count <= 0:
        return 30, "About 30 seconds"
    seconds = min(180, max(30, card_count * 4))
    minutes = seconds // 60
    if minutes:
        return seconds, f"Roughly {minutes} minute{'s' if minutes > 1 else ''}"
    return seconds, f"About {seconds} seconds"


def estimate_media_duration(card_count: int, per_card_seconds: float) -> Tuple[int, str]:
    if card_count <= 0:
        return 60, "About 1 minute"
    seconds = int(min(1800, max(45, card_count * per_card_seconds)))
    minutes = seconds // 60
    if minutes:
        return seconds, f"Roughly {minutes} minute{'s' if minutes > 1 else ''}"
    return seconds, f"About {seconds} seconds"


def get_deck_card_count(deckname: str) -> int:
    try:
        # Quote deck name to handle spaces/special characters like '&'
        cards = invoke("findNotes", query=f'deck:"{deckname}"')
        return len(cards)
    except Exception:
        return 0


def clean_field_text(raw: str) -> str:
    raw = raw or ""
    without_sound = SOUND_TAG_RE.sub(" ", raw)
    without_nbsp = NBSP_RE.sub(" ", without_sound)
    without_tags = HTML_TAG_RE.sub(" ", without_nbsp)
    return " ".join(without_tags.split())


def extract_image_filename(html: str) -> str:
    match = IMG_SRC_RE.search(html or "")
    if match:
        return Path(match.group(1)).name
    return ""


@app.route("/sync", methods=["POST"])
def sync_deck():
    uploaded_file = request.files.get("file")
    deck_name = request.form.get("deck", "").strip()
    model = request.form.get("model", "").strip() or "gpt-4.1-mini"
    include_romanized = request.form.get("romanized", "true") == "true"

    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"ok": False, "message": "No PDF uploaded."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"ok": False, "message": "Only PDF files are supported."}), 400

    safe_name = secure_filename(uploaded_file.filename)
    saved_path = UPLOAD_DIR / safe_name
    uploaded_file.save(saved_path)

    command_args = [str(saved_path)]
    if deck_name:
        command_args.extend(["--deck", deck_name])
    if model:
        command_args.extend(["--model", model])
    if include_romanized:
        command_args.append("--romanized")
    else:
        command_args.append("--no-romanized")

    try:
        result = run_script(BASE_DIR / "AnkiSync.py", command_args)
        deck_for_estimate = deck_name or safe_name.rsplit(".", 1)[0]
        count = get_deck_card_count(deck_for_estimate)
        eta_seconds, eta_text = estimate_sync_duration(count)
        return jsonify(
            {
                "ok": True,
                "message": "Deck synced successfully.",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "eta_seconds": eta_seconds,
                "eta_text": eta_text,
                "items_processed": count,
            }
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "AnkiSync failed.",
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                }
            ),
            500,
        )
    except RuntimeError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/generate/audio", methods=["POST"])
def generate_audio():
    data = request.get_json(silent=True) or {}
    deck = data.get("deck", "").strip()
    if not deck:
        return jsonify({"ok": False, "message": "Deck name is required."}), 400

    args = [deck]
    model = data.get("model")
    voice = data.get("voice")
    workers = data.get("workers")
    instructions = data.get("instructions")

    if model:
        args.extend(["--model", model])
    if voice:
        args.extend(["--voice", voice])
    if instructions:
        args.extend(["--instructions", instructions])
    if workers:
        args.extend(["--workers", str(workers)])

    try:
        result = run_script(BASE_DIR / "AnkiDeckToSpeech.py", args)
        card_count = get_deck_card_count(deck)
        eta_seconds, eta_text = estimate_media_duration(card_count, per_card_seconds=6.0)
        return jsonify(
            {
                "ok": True,
                "message": "Audio generation completed.",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "eta_seconds": eta_seconds,
                "eta_text": eta_text,
                "items_processed": card_count,
            }
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "Audio generation failed.",
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                }
            ),
            500,
        )
    except RuntimeError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/generate/images", methods=["POST"])
def generate_images():
    data = request.get_json(silent=True) or {}
    deck = data.get("deck", "").strip()
    if not deck:
        return jsonify({"ok": False, "message": "Deck name is required."}), 400

    args = [deck]
    image_model = data.get("image_model")
    prompt = data.get("prompt")
    workers = data.get("workers")
    skip_gating = data.get("skip_gating", False)

    if image_model:
        args.extend(["--image-model", image_model])
    if prompt:
        args.extend(["--prompt", prompt])
    if workers:
        args.extend(["--workers", str(workers)])
    if skip_gating:
        args.append("--skip-gating")

    try:
        result = run_script(BASE_DIR / "AnkiDeckToImages.py", args)
        card_count = get_deck_card_count(deck)
        eta_seconds, eta_text = estimate_media_duration(card_count, per_card_seconds=12.0)
        return jsonify(
            {
                "ok": True,
                "message": "Image generation completed.",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "eta_seconds": eta_seconds,
                "eta_text": eta_text,
                "items_processed": card_count,
            }
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "Image generation failed.",
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                }
            ),
            500,
        )
    except RuntimeError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/media/images/<path:filename>")
def serve_image_file(filename: str):
    target = IMAGE_DIR / filename
    if not target.exists():
        return jsonify({"ok": False, "message": "Image not found."}), 404
    return send_from_directory(IMAGE_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
