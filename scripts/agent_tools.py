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


ROOT = Path(__file__).resolve().parents[1]
TOOLS_MANIFEST = ROOT / "agent_tools.json"


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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
