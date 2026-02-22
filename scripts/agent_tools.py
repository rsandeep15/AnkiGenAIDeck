#!/usr/bin/env python3
"""
Agent-friendly wrapper around canonical project workflows.

This is intentionally thin: it routes to existing scripts so agents can discover
and run standard flows before writing bespoke AnkiConnect logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS_MANIFEST = ROOT / "agent_tools.json"
LOCAL_DOCS_DIR = ROOT / ".local_memory" / "pdfs"


def run_script(script_name: str, args: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / script_name), *args]
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_list(_: argparse.Namespace) -> int:
    if TOOLS_MANIFEST.exists():
        print(TOOLS_MANIFEST.read_text(encoding="utf-8"))
        return 0

    fallback = {
        "entrypoint": "scripts/agent_tools.py",
        "tools": [
            {"id": "sync_pdf_to_deck"},
            {"id": "generate_audio_for_deck"},
            {"id": "generate_images_for_deck"},
            {"id": "import_cards_to_deck"},
            {"id": "evaluate_image_gating"},
            {"id": "hillclimb_image_gating"},
            {"id": "upload_image_gating_dataset"},
            {"id": "list_local_reference_pdfs"},
            {"id": "read_local_reference_pdf"},
            {"id": "run_web_ui"},
        ],
    }
    print(json.dumps(fallback, ensure_ascii=False, indent=2))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    cmd_args = [str(args.pdf), "--deck", args.deck, "--model", args.model]
    cmd_args.append("--romanized" if args.romanized else "--no-romanized")
    if args.note_model:
        cmd_args.extend(["--note-model", args.note_model])
    if args.dry_run:
        cmd_args.append("--dry-run")
    return run_script("AnkiSync.py", cmd_args)


def cmd_audio(args: argparse.Namespace) -> int:
    cmd_args = [
        args.deck,
        "--model",
        args.model,
        "--voice",
        args.voice,
        "--workers",
        str(args.workers),
    ]
    if args.instructions:
        cmd_args.extend(["--instructions", args.instructions])
    if args.dry_run:
        cmd_args.append("--dry-run")
    return run_script("AnkiDeckToSpeech.py", cmd_args)


def cmd_images(args: argparse.Namespace) -> int:
    cmd_args = [
        args.deck,
        "--image-model",
        args.image_model,
        "--workers",
        str(args.workers),
    ]
    if args.prompt:
        cmd_args.extend(["--prompt", args.prompt])
    if args.limit > 0:
        cmd_args.extend(["--limit", str(args.limit)])
    if args.shuffle:
        cmd_args.extend(["--shuffle", "--seed", str(args.seed)])
    if args.dry_run:
        cmd_args.append("--dry-run")
    return run_script("AnkiDeckToImages.py", cmd_args)


def cmd_ui(args: argparse.Namespace) -> int:
    port = str(args.port)
    cmd = [sys.executable, "-m", "flask", "--app", "app.py", "run", "--port", port]
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_gating(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "image_gating_eval.py"), *args.gating_args]
    return subprocess.run(cmd, cwd=ROOT).returncode


def resolve_local_docs_dir(custom_dir: str | None = None) -> Path:
    path = Path(custom_dir).expanduser() if custom_dir else LOCAL_DOCS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_docs_list(args: argparse.Namespace) -> int:
    docs_dir = resolve_local_docs_dir(args.dir)
    files = sorted(docs_dir.rglob("*.pdf"))
    payload = {
        "ok": True,
        "docs_dir": str(docs_dir),
        "count": len(files),
        "files": [
            {
                "name": p.name,
                "relative_path": str(p.relative_to(docs_dir)),
                "bytes": p.stat().st_size,
                "modified_epoch": int(p.stat().st_mtime),
            }
            for p in files
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_docs_read(args: argparse.Namespace) -> int:
    docs_dir = resolve_local_docs_dir(args.dir)
    target = (docs_dir / args.file).resolve()
    try:
        target.relative_to(docs_dir.resolve())
    except ValueError:
        print("File path must stay inside the local docs directory.")
        return 1
    if not target.exists():
        print(f"PDF not found: {target}")
        return 1

    from pypdf import PdfReader

    reader = PdfReader(str(target))
    pages = len(reader.pages)
    page_num = args.page
    if page_num < 1 or page_num > pages:
        print(f"Page out of range. PDF has {pages} page(s).")
        return 1
    text = reader.pages[page_num - 1].extract_text() or ""
    text = text.strip()
    if args.max_chars > 0:
        text = text[: args.max_chars]
    payload = {
        "ok": True,
        "file": str(target),
        "page": page_num,
        "pages": pages,
        "text": text,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def load_card_rows(input_path: Path) -> list[dict[str, Any]]:
    content = input_path.read_text(encoding="utf-8")
    payload = json.loads(content)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cards"), list):
        rows = payload["cards"]
    else:
        raise ValueError(
            "Input JSON must be a list of card objects or an object with a 'cards' list."
        )

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
    return normalized_rows


def _first_nonempty_value(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_cards(
    rows: list[dict[str, Any]], front_key: str, back_key: str
) -> list[tuple[str, str]]:
    cards: list[tuple[str, str]] = []
    for row in rows:
        front = _first_nonempty_value(
            row,
            [
                front_key,
                front_key.lower(),
                front_key.capitalize(),
                "Front",
                "front",
                "foreign",
                "korean",
            ],
        )
        back = _first_nonempty_value(
            row,
            [
                back_key,
                back_key.lower(),
                back_key.capitalize(),
                "Back",
                "back",
                "english",
                "meaning",
            ],
        )
        if front and back:
            cards.append((front, back))
    return cards


def cmd_cards_import(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    try:
        rows = load_card_rows(input_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to load cards from {input_path}: {exc}")
        return 1

    pairs = normalize_cards(rows, args.front_key, args.back_key)
    if not pairs:
        print(
            "No valid cards found. Ensure rows contain front/back values "
            f"for keys '{args.front_key}' and '{args.back_key}'."
        )
        return 1

    deduped_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if not args.allow_duplicates and pair in seen:
            continue
        deduped_pairs.append(pair)
        seen.add(pair)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "deck": args.deck,
                    "input_rows": len(rows),
                    "valid_pairs": len(pairs),
                    "to_add": len(deduped_pairs),
                    "sample": [
                        {"Front": f, "Back": b} for f, b in deduped_pairs[:5]
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils.anki_connect import invoke

    invoke("createDeck", deck=args.deck)
    notes = [
        {
            "deckName": args.deck,
            "modelName": args.note_model,
            "fields": {"Front": front, "Back": back},
        }
        for front, back in deduped_pairs
    ]
    result = invoke("addNotes", notes=notes) or []
    added = len([note_id for note_id in result if note_id])
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": False,
                "deck": args.deck,
                "input_rows": len(rows),
                "valid_pairs": len(pairs),
                "attempted_add": len(deduped_pairs),
                "added": added,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent-first wrapper for canonical Anki toolkit flows."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s_list = sub.add_parser("list", help="Print the machine-readable tool manifest.")
    s_list.set_defaults(func=cmd_list)

    s_sync = sub.add_parser("sync", help="Sync a PDF into an Anki deck.")
    s_sync.add_argument("--pdf", type=Path, required=True, help="Path to source PDF.")
    s_sync.add_argument("--deck", required=True, help="Target deck name.")
    s_sync.add_argument("--model", default="gpt-4.1-mini")
    s_sync.add_argument("--note-model", default="Basic")
    s_sync.add_argument("--romanized", action="store_true")
    s_sync.add_argument("--dry-run", action="store_true")
    s_sync.set_defaults(func=cmd_sync)

    s_audio = sub.add_parser("audio", help="Generate audio for a deck.")
    s_audio.add_argument("--deck", required=True)
    s_audio.add_argument("--model", default="gpt-4o-mini-tts")
    s_audio.add_argument("--voice", default="onyx")
    s_audio.add_argument("--workers", type=int, default=10)
    s_audio.add_argument("--instructions", default="")
    s_audio.add_argument("--dry-run", action="store_true")
    s_audio.set_defaults(func=cmd_audio)

    s_images = sub.add_parser("images", help="Generate images for a deck.")
    s_images.add_argument("--deck", required=True)
    s_images.add_argument("--image-model", default="gpt-image-1")
    s_images.add_argument("--workers", type=int, default=3)
    s_images.add_argument("--prompt", default="")
    s_images.add_argument("--limit", type=int, default=0)
    s_images.add_argument("--shuffle", action="store_true")
    s_images.add_argument("--seed", type=int, default=42)
    s_images.add_argument("--dry-run", action="store_true")
    s_images.set_defaults(func=cmd_images)

    s_ui = sub.add_parser("ui", help="Run Flask UI.")
    s_ui.add_argument("--port", type=int, default=5000)
    s_ui.set_defaults(func=cmd_ui)

    s_docs_list = sub.add_parser(
        "docs-list",
        help="List local reference PDFs from the local memory directory.",
    )
    s_docs_list.add_argument(
        "--dir",
        default="",
        help="Optional docs directory override (default: .local_memory/pdfs).",
    )
    s_docs_list.set_defaults(func=cmd_docs_list)

    s_docs_read = sub.add_parser(
        "docs-read",
        help="Read extracted text from a specific page of a local reference PDF.",
    )
    s_docs_read.add_argument("--file", required=True, help="PDF filename or relative path.")
    s_docs_read.add_argument("--page", type=int, default=1, help="1-based page number.")
    s_docs_read.add_argument(
        "--max-chars", type=int, default=4000, help="Max characters returned."
    )
    s_docs_read.add_argument(
        "--dir",
        default="",
        help="Optional docs directory override (default: .local_memory/pdfs).",
    )
    s_docs_read.set_defaults(func=cmd_docs_read)

    s_gating = sub.add_parser(
        "gating",
        help="Run image gating tools (make-template/eval/hillclimb/upload-dataset).",
    )
    s_gating.add_argument("gating_args", nargs=argparse.REMAINDER)
    s_gating.set_defaults(func=cmd_gating)

    s_cards = sub.add_parser(
        "cards-import",
        help="Import explicit Front/Back card pairs from JSON into a deck.",
    )
    s_cards.add_argument("--deck", required=True, help="Target deck name.")
    s_cards.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to JSON file (list of objects or {'cards': [...]}).",
    )
    s_cards.add_argument(
        "--front-key",
        default="Front",
        help="Source key for front text in each row (default: Front).",
    )
    s_cards.add_argument(
        "--back-key",
        default="Back",
        help="Source key for back text in each row (default: Back).",
    )
    s_cards.add_argument(
        "--note-model",
        default="Basic",
        help="Anki note model to use (default: Basic).",
    )
    s_cards.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Do not deduplicate identical Front/Back pairs within this import.",
    )
    s_cards.add_argument("--dry-run", action="store_true")
    s_cards.set_defaults(func=cmd_cards_import)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
