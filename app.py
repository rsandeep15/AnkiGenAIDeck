import base64
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)
from werkzeug.utils import secure_filename

from openai import OpenAI
from utils.anki_connect import invoke
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
PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d+)\s*/\s*(\d+)\s*$")
SUMMARY_RE = re.compile(r"^SUMMARY:\s*(\{.*\})\s*$")


@dataclass
class Job:
    job_id: str
    kind: str
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    exit_code: int | None = None
    summary: dict | None = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def parse_progress_line(line: str) -> tuple[int, int] | None:
    match = PROGRESS_RE.match(line.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_summary_line(line: str) -> dict | None:
    match = SUMMARY_RE.match(line.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def enqueue_event(job: Job, event: str, data: dict) -> None:
    job.events.put({"event": event, "data": data})


def run_script_stream(job: Job, script_path: Path, args: List[str]) -> None:
    env = os.environ.copy()
    if "OPENAI_API_KEY" not in env:
        job.status = "failed"
        enqueue_event(job, "log", {"message": "OPENAI_API_KEY is not set on the server."})
        enqueue_event(job, "done", {"ok": False, "exit_code": None, "summary": None})
        return

    command = [sys.executable, str(script_path)] + args
    job.status = "running"
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
    except Exception as exc:
        job.status = "failed"
        enqueue_event(job, "log", {"message": f"Failed to start process: {exc}"})
        enqueue_event(job, "done", {"ok": False, "exit_code": None, "summary": None})
        return

    if process.stdout is None:
        job.status = "failed"
        enqueue_event(job, "log", {"message": "No stdout available for streaming."})
        enqueue_event(job, "done", {"ok": False, "exit_code": None, "summary": None})
        return

    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if line:
            progress = parse_progress_line(line)
            if progress:
                current, total = progress
                enqueue_event(job, "progress", {"current": current, "total": total})
                continue
            enqueue_event(job, "log", {"message": line})
            summary = parse_summary_line(line)
            if summary:
                job.summary = summary

    process.wait()
    job.exit_code = process.returncode
    job.status = "completed" if process.returncode == 0 else "failed"
    enqueue_event(
        job,
        "done",
        {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "summary": job.summary,
        },
    )


def launch_job(kind: str, script_path: Path, args: List[str]) -> Job:
    job_id = uuid.uuid4().hex
    job = Job(job_id=job_id, kind=kind)
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(
        target=run_script_stream,
        args=(job, script_path, args),
        daemon=True,
    )
    thread.start()
    return job


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
                    "sound_filename": extract_sound_filename(front) or extract_sound_filename(back),
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


@app.route("/api/media", methods=["GET"])
def media_file():
    filename = request.args.get("filename", "").strip()
    if not filename:
        return jsonify({"ok": False, "message": "filename parameter is required."}), 400
    try:
        data = invoke("retrieveMediaFile", filename=filename)
        if not data:
            return jsonify({"ok": False, "message": "Media not found."}), 404
        payload = base64.b64decode(data)
        if filename.lower().endswith(".mp3"):
            mimetype = "audio/mpeg"
        elif filename.lower().endswith(".wav"):
            mimetype = "audio/wav"
        elif filename.lower().endswith(".ogg"):
            mimetype = "audio/ogg"
        else:
            mimetype = "application/octet-stream"
        return app.response_class(payload, mimetype=mimetype)
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/api/deck-audio-stats", methods=["GET"])
def deck_audio_stats():
    deck = request.args.get("deck", "").strip()
    if not deck:
        return jsonify({"ok": False, "message": "Deck parameter is required."}), 400

    try:
        total_notes = invoke("findNotes", query=f'deck:"{deck}"') or []
        audio_notes = set(
            invoke("findNotes", query=f'deck:"{deck}" "sound:"') or []
        )
        if not audio_notes:
            # Fallback to explicit field search if needed.
            audio_notes.update(
                invoke("findNotes", query=f'deck:"{deck}" front:*[sound:*') or []
            )
            audio_notes.update(
                invoke("findNotes", query=f'deck:"{deck}" back:*[sound:*') or []
            )
        total = len(total_notes)
        with_audio = len(audio_notes)
        return jsonify(
            {
                "ok": True,
                "deck": deck,
                "total": total,
                "with_audio": with_audio,
                "coverage": round((with_audio / total) * 100, 1) if total else 0.0,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/api/deck-image-stats", methods=["GET"])
def deck_image_stats():
    deck = request.args.get("deck", "").strip()
    if not deck:
        return jsonify({"ok": False, "message": "Deck parameter is required."}), 400

    try:
        total_notes = invoke("findNotes", query=f'deck:"{deck}"') or []
        image_notes = set(
            invoke("findNotes", query=f'deck:"{deck}" "image:"') or []
        )
        if not image_notes:
            image_notes.update(
                invoke("findNotes", query=f'deck:"{deck}" front:*<img*') or []
            )
            image_notes.update(
                invoke("findNotes", query=f'deck:"{deck}" back:*<img*') or []
            )
        total = len(total_notes)
        with_images = len(image_notes)
        return jsonify(
            {
                "ok": True,
                "deck": deck,
                "total": total,
                "with_images": with_images,
                "coverage": round((with_images / total) * 100, 1) if total else 0.0,
            }
        )
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


def extract_sound_filename(text: str) -> str:
    match = SOUND_TAG_RE.search(text or "")
    if not match:
        return ""
    raw = match.group(0)
    if raw.startswith("[sound:") and raw.endswith("]"):
        return raw[len("[sound:"):-1]
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


@app.route("/api/jobs/audio", methods=["POST"])
def start_audio_job():
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

    job = launch_job("audio", BASE_DIR / "AnkiDeckToSpeech.py", args)
    return jsonify(
        {
            "ok": True,
            "job_id": job.job_id,
            "stream_url": url_for("stream_job", job_id=job.job_id),
        }
    )


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


@app.route("/api/jobs/images", methods=["POST"])
def start_image_job():
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

    job = launch_job("images", BASE_DIR / "AnkiDeckToImages.py", args)
    return jsonify(
        {
            "ok": True,
            "job_id": job.job_id,
            "stream_url": url_for("stream_job", job_id=job.job_id),
        }
    )


@app.route("/api/jobs/<job_id>/stream", methods=["GET"])
def stream_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "message": "Job not found."}), 404

    def generate():
        yield "retry: 1000\n\n"
        while True:
            try:
                event = job.events.get(timeout=1.0)
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\n"
                yield f"data: {payload}\n\n"
            except queue.Empty:
                if job.status in {"completed", "failed"} and job.events.empty():
                    break
                yield ":\n\n"

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/media/images/<path:filename>")
def serve_image_file(filename: str):
    target = IMAGE_DIR / filename
    if not target.exists():
        return jsonify({"ok": False, "message": "Image not found."}), 404
    return send_from_directory(IMAGE_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
