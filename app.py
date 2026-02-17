import base64
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
from config import OPENAI_STORE_RESPONSES
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
app.instance_path = os.path.join(BASE_DIR, "instance")
os.makedirs(app.instance_path, exist_ok=True)
PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d+)\s*/\s*(\d+)\s*$")
SUMMARY_RE = re.compile(r"^SUMMARY:\s*(\{.*\})\s*$")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
SEARCH_TERM_RE = re.compile(r"[\n\r\t]+")


@dataclass
class Job:
    job_id: str
    kind: str
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    exit_code: Optional[int] = None
    summary: Optional[dict] = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def parse_progress_line(line: str) -> Optional[Tuple[int, int]]:
    match = PROGRESS_RE.match(line.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_summary_line(line: str) -> Optional[dict]:
    match = SUMMARY_RE.match(line.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def enqueue_event(job: Job, event: str, data: dict) -> None:
    job.events.put({"event": event, "data": data})


def get_chat_db_path() -> str:
    return os.path.join(app.instance_path, "chat_logs.sqlite")


def init_chat_db() -> None:
    path = get_chat_db_path()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                deck TEXT NOT NULL,
                model TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                history_json TEXT
            )
            """
        )
        conn.commit()


def log_chat_event(
    *, deck: str, model: Optional[str], question: str, answer: str, history: Optional[list]
) -> None:
    init_chat_db()
    payload = json.dumps(history or [], ensure_ascii=False)
    with sqlite3.connect(get_chat_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO chat_logs (created_at, deck, model, question, answer, history_json)
            VALUES (datetime('now'), ?, ?, ?, ?, ?)
            """,
            (deck, model, question, answer, payload),
        )
        conn.commit()


@app.route("/api/chat-logs/export", methods=["GET"])
def export_chat_logs():
    fmt = (request.args.get("format") or "jsonl").lower()
    limit = request.args.get("limit")
    try:
        limit_val = int(limit) if limit else None
    except ValueError:
        return jsonify({"ok": False, "message": "Invalid limit."}), 400

    init_chat_db()
    with sqlite3.connect(get_chat_db_path()) as conn:
        cursor = conn.cursor()
        if limit_val:
            cursor.execute(
                "SELECT created_at, deck, model, question, answer, history_json FROM chat_logs ORDER BY id DESC LIMIT ?",
                (limit_val,),
            )
        else:
            cursor.execute(
                "SELECT created_at, deck, model, question, answer, history_json FROM chat_logs ORDER BY id DESC"
            )
        rows = cursor.fetchall()

    if fmt == "csv":
        output = ["created_at,deck,model,question,answer,history_json"]
        for created_at, deck, model, question, answer, history_json in rows:
            output.append(
                ",".join(
                    [
                        json.dumps(created_at, ensure_ascii=False),
                        json.dumps(deck, ensure_ascii=False),
                        json.dumps(model or "", ensure_ascii=False),
                        json.dumps(question, ensure_ascii=False),
                        json.dumps(answer, ensure_ascii=False),
                        json.dumps(history_json or "", ensure_ascii=False),
                    ]
                )
            )
        return Response("\n".join(output), mimetype="text/csv")

    lines = []
    for created_at, deck, model, question, answer, history_json in rows:
        record = {
            "created_at": created_at,
            "deck": deck,
            "model": model,
            "question": question,
            "answer": answer,
            "history": json.loads(history_json) if history_json else [],
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return Response("\n".join(lines), mimetype="application/jsonl")


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


def is_text_model(model_id: str) -> bool:
    blocked = ("tts", "audio", "image", "embed", "embedding", "speech")
    if not model_id.startswith(("gpt", "o")):
        return False
    return not any(token in model_id for token in blocked)


def is_audio_model(model_id: str) -> bool:
    return "tts" in model_id or model_id.endswith("-tts") or "audio" in model_id


def is_image_model(model_id: str) -> bool:
    return "image" in model_id or model_id.startswith("dall-e")


def filter_models(kind: str) -> List[str]:
    kind = kind.lower()
    ids = cached_model_ids()
    popular_text = [
        "gpt-5.2",
        "gpt-5.2-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    if kind == "text":
        latest_like = [
            model_id
            for model_id in ids
            if (
                model_id.startswith("gpt-5")
                or model_id.endswith("-latest")
                or model_id.endswith("-latest-preview")
            )
            and is_text_model(model_id)
        ]
        ordered = []
        for model_id in popular_text + latest_like:
            if model_id in ids and model_id not in ordered:
                ordered.append(model_id)
        available = ordered
        if available:
            return available
        return sorted(filter(is_text_model, ids))
    if kind == "audio":
        return sorted(filter(is_audio_model, ids))
    if kind == "image":
        return sorted(filter(is_image_model, ids))
    raise ValueError("Unsupported model kind")


def get_response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text
    output = getattr(resp, "output", None)
    if not output:
        return ""
    parts: List[str] = []
    for item in output:
        for content_piece in getattr(item, "content", []):
            maybe_text = getattr(content_piece, "text", None)
            if maybe_text:
                parts.append(maybe_text)
    return "".join(parts)


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


def select_chat_model() -> str:
    try:
        models = filter_models("text")
        if models:
            return models[0]
    except Exception:
        pass
    return "gpt-4.1-mini"


def tokenize_text(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def build_deck_context_full(deck: str) -> Tuple[List[str], int]:
    note_ids = invoke("findNotes", query=f'deck:"{deck}"') or []
    if not note_ids:
        return [], 0
    notes = invoke("notesInfo", notes=note_ids)
    snippets = []
    for note in notes:
        fields = note.get("fields", {})
        front = clean_field_text((fields.get("Front") or {}).get("value", ""))
        back = clean_field_text((fields.get("Back") or {}).get("value", ""))
        if not front and not back:
            continue
        combined = f"{front} — {back}".strip(" —")
        if not combined:
            continue
        snippets.append(combined)
    return snippets, len(note_ids)


@app.route("/api/deck-images", methods=["GET"])
def deck_images():
    deck = request.args.get("deck", "").strip()
    if not deck:
        return jsonify({"ok": False, "message": "Deck parameter is required."}), 400
    try:
        page = int(request.args.get("page", "1"))
        page_size = int(request.args.get("page_size", "24"))
    except ValueError:
        return jsonify({"ok": False, "message": "Invalid page or page_size."}), 400
    if page < 1 or page_size < 1 or page_size > 200:
        return jsonify({"ok": False, "message": "Invalid page or page_size."}), 400

    try:
        note_ids = invoke("findNotes", query=f'deck:"{deck}"')
        if not note_ids:
            return jsonify(
                {
                    "ok": True,
                    "images": [],
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                }
            )
        notes = invoke("notesInfo", notes=note_ids)
        results = []
        for note_id, note in zip(note_ids, notes):
            front = note["fields"]["Front"]["value"]
            back = note["fields"]["Back"]["value"]
            front_image = extract_image_filename(front)
            back_image = extract_image_filename(back)
            filename = front_image or back_image
            if not filename:
                continue
            image_side = "Front" if front_image else "Back"
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
                    "back_text": clean_field_text(back),
                    "image_side": image_side,
                    "sound_filename": extract_sound_filename(front) or extract_sound_filename(back),
                    "image_url": url_for("serve_image_file", filename=local_path.name),
                }
            )
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        return jsonify(
            {
                "ok": True,
                "images": results[start:end],
                "page": page,
                "page_size": page_size,
                "total": total,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/api/deck-gallery", methods=["GET"])
def deck_gallery():
    deck = request.args.get("deck", "").strip()
    term = normalize_search_term(request.args.get("term", ""))
    if not deck:
        return jsonify({"ok": False, "message": "Deck parameter is required."}), 400
    try:
        page = int(request.args.get("page", "1"))
        page_size = int(request.args.get("page_size", "24"))
    except ValueError:
        return jsonify({"ok": False, "message": "Invalid page or page_size."}), 400
    if page < 1 or page_size < 1 or page_size > 200:
        return jsonify({"ok": False, "message": "Invalid page or page_size."}), 400

    try:
        query = f'deck:"{deck}"'
        if term:
            query = (
                f'deck:"{deck}" '
                f'({build_field_query("Front", term)} OR {build_field_query("Back", term)})'
            )
        note_ids = invoke("findNotes", query=query) or []
        if not note_ids:
            return jsonify(
                {
                    "ok": True,
                    "items": [],
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "term": term,
                }
            )

        notes = invoke("notesInfo", notes=note_ids)
        results = []
        for note_id, note in zip(note_ids, notes):
            fields = note.get("fields", {})
            front_raw = (fields.get("Front") or {}).get("value", "")
            back_raw = (fields.get("Back") or {}).get("value", "")
            front_image = extract_image_filename(front_raw)
            back_image = extract_image_filename(back_raw)
            filename = front_image or back_image
            image_url = ""
            if filename:
                local_path = IMAGE_DIR / filename
                if not local_path.exists():
                    stem, suffix = os.path.splitext(filename)
                    if "-" in stem:
                        base = stem.split("-", 1)[0] + suffix
                        alt_path = IMAGE_DIR / base
                        if alt_path.exists():
                            local_path = alt_path
                        else:
                            local_path = None
                    else:
                        local_path = None
                if local_path and local_path.exists():
                    image_url = url_for("serve_image_file", filename=local_path.name)

            results.append(
                {
                    "id": note_id,
                    "front_text": clean_field_text(front_raw),
                    "back_text": clean_field_text(back_raw),
                    "has_image": bool(image_url),
                    "image_url": image_url,
                    "image_side": "Front" if front_image else ("Back" if back_image else ""),
                    "sound_filename": extract_sound_filename(front_raw) or extract_sound_filename(back_raw),
                }
            )

        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        return jsonify(
            {
                "ok": True,
                "items": results[start:end],
                "page": page,
                "page_size": page_size,
                "total": total,
                "term": term,
            }
        )
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


@app.route("/api/deck-search", methods=["GET"])
def deck_search():
    term = request.args.get("term", "")
    limit_raw = request.args.get("limit", "50")
    term = normalize_search_term(term)
    if not term:
        return jsonify({"ok": False, "message": "Search term is required."}), 400

    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(200, limit))

    try:
        query = f"({build_field_query('Front', term)} OR {build_field_query('Back', term)})"
        card_ids = invoke("findCards", query=query) or []
        if not card_ids:
            return jsonify({"ok": True, "results": []})

        card_ids = card_ids[:limit]
        cards = invoke("cardsInfo", cards=card_ids) or []
        deck_by_note: dict[int, str] = {}
        note_ids: list[int] = []
        for card in cards:
            note_id = card.get("note")
            if not note_id:
                continue
            if note_id not in deck_by_note:
                deck_by_note[note_id] = card.get("deckName") or "Unknown"
            note_ids.append(note_id)

        unique_note_ids = list(dict.fromkeys(note_ids))
        notes = invoke("notesInfo", notes=unique_note_ids) if unique_note_ids else []
        results = []
        term_lower = term.lower()
        for note_id, note in zip(unique_note_ids, notes):
            fields = note.get("fields", {})
            front = clean_field_text((fields.get("Front") or {}).get("value", ""))
            back = clean_field_text((fields.get("Back") or {}).get("value", ""))
            romanized = clean_field_text((fields.get("Romanized") or {}).get("value", ""))
            deck_name = deck_by_note.get(note_id, "Unknown")
            in_front = term_lower in front.lower()
            in_back = term_lower in back.lower()
            if in_front and in_back:
                match = "both"
            elif in_front:
                match = "front"
            elif in_back:
                match = "back"
            else:
                match = ""
            results.append(
                {
                    "id": note_id,
                    "deck": deck_name,
                    "front": front,
                    "back": back,
                    "romanized": romanized,
                    "match": match,
                }
            )

        return jsonify({"ok": True, "results": results})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.route("/api/deck-chat", methods=["POST"])
def deck_chat():
    data = request.get_json(silent=True) or {}
    deck = data.get("deck", "").strip()
    question = data.get("question", "").strip()
    stream = bool(data.get("stream"))
    history = data.get("history") or []
    requested_model = (data.get("model") or "").strip()
    if not deck or not question:
        return jsonify({"ok": False, "message": "Deck and question are required."}), 400

    try:
        context_snippets, total_cards = build_deck_context_full(deck)
        if not context_snippets:
            return jsonify(
                {
                    "ok": True,
                    "answer": "I couldn't find any cards in that deck to reference.",
                    "model": None,
                    "snippets": [],
                }
            )
        client = OpenAI()
        model = select_chat_model()
        if requested_model:
            try:
                if requested_model in cached_model_ids() and is_text_model(requested_model):
                    model = requested_model
            except Exception:
                pass
        context_block = "\n".join(f"- {snippet}" for snippet in context_snippets)
        history_block = ""
        if isinstance(history, list):
            trimmed = history[-6:]
            history_lines = []
            for item in trimmed:
                role = (item.get("role") or "user").strip().lower()
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                label = "User" if role == "user" else "Assistant"
                history_lines.append(f"{label}: {content}")
            if history_lines:
                history_block = "\n".join(history_lines)
        convo_block = ""
        if history_block:
            convo_block = f"Conversation so far:\n{history_block}\n\n"
        prompt = (
            "You are a study assistant for an Anki vocabulary deck.\n"
            "You are given the full list of card text only (no audio or images).\n"
            "Use the deck contents as a starting point, but you may add helpful external knowledge\n"
            "to explain usage, nuance, and examples. Do not fabricate cards that are not in the deck.\n"
            "Be concise and tutoring-focused: 2-4 sentences max unless asked for more.\n"
            "Prefer a short explanation + 3 bullet examples max (only if they help).\n"
            "If the question is broad, ask 1 clarifying question instead of dumping info.\n\n"
            "The user's native language is English; respond primarily in English and include foreign words only when helpful.\n\n"
            "Use plain text only. Avoid markdown or formatting symbols like **, _, `, #, >.\n\n"
            "If the user asks for a quiz, ask ONE item at a time and wait for their reply.\n"
            "Use simple bullets for lists. Avoid numbered lists unless the user requests them.\n\n"
            "If you provide examples, give at most 3 and offer to continue if needed.\n\n"
            f"Deck: {deck} (total cards: {total_cards})\n"
            f"Deck word pairs:\n{context_block}\n\n"
            f"{convo_block}"
            f"Question: {question}\nAnswer:"
        )
        reasoning_opts = None
        text_opts = None
        if model.startswith("gpt-5"):
            reasoning_opts = {"effort": "medium"}
            text_opts = {"verbosity": "low"}
        metadata = {
            "app": "anki_deck_studio",
            "feature": "deck_chat",
            "deck": deck,
        }

        if stream:
            def generate():
                streamed_text = []
                response_stream = client.responses.create(
                    model=model,
                    input=prompt,
                    reasoning=reasoning_opts,
                    text=text_opts,
                    store=OPENAI_STORE_RESPONSES,
                    metadata=metadata,
                    stream=True,
                )
                for event in response_stream:
                    delta = getattr(event, "delta", None)
                    if not delta:
                        delta = getattr(event, "output_text", None)
                    if not delta:
                        delta = getattr(event, "text", None)
                    if delta:
                        streamed_text.append(delta)
                        yield delta
                final_answer = "".join(streamed_text).strip()
                if final_answer:
                    log_chat_event(
                        deck=deck,
                        model=model,
                        question=question,
                        answer=final_answer,
                        history=history,
                    )
            response = Response(stream_with_context(generate()), mimetype="text/plain")
            response.headers["X-Chat-Model"] = model
            return response
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning=reasoning_opts,
            text=text_opts,
            store=OPENAI_STORE_RESPONSES,
            metadata=metadata,
        )
        answer = get_response_text(response).strip() or "No response."
        log_chat_event(
            deck=deck,
            model=model,
            question=question,
            answer=answer,
            history=history,
        )
        return jsonify({"ok": True, "answer": answer, "model": model, "snippets": context_snippets})
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


def normalize_search_term(term: str) -> str:
    cleaned = (term or "").strip()
    cleaned = SEARCH_TERM_RE.sub(" ", cleaned)
    cleaned = cleaned.replace('"', "").replace("'", "")
    return " ".join(cleaned.split())


def build_field_query(field: str, term: str) -> str:
    if " " in term:
        return f'{field}:"{term}"'
    return f"{field}:*{term}*"


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
