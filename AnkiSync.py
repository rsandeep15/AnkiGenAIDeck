import argparse
import json
import os
from pathlib import Path
import sys
import shutil
from typing import Any, Dict, List

from openai import OpenAI

from config import DEFAULT_TEXT_MODEL
from utils.common import collapse_whitespace, strip_english_duplicate
from utils.anki_connect import invoke

# Function to create a file with the Files API
def create_file(client: OpenAI, file_path: Path) -> str:
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="assistants",
        )
    return result.id


def request(action: str, **params: Any) -> Dict[str, Any]:
    return {"action": action, "params": params, "version": 6}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a PDF of vocabulary pairs into an Anki deck using AnkiConnect."
    )
    parser.add_argument("pdf", type=Path, help="Path to the source PDF file.")
    parser.add_argument(
        "--deck",
        help="Name of the deck to create (defaults to the PDF filename without extension).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TEXT_MODEL,
        help=(
            "OpenAI model used to extract vocabulary (default: %(default)s). "
            "Try faster tiers like 'gpt-4o-mini' or higher-accuracy models like 'gpt-4.1'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the PDF and print a summary without writing to Anki.",
    )
    parser.add_argument(
        "--note-model",
        default="Basic",
        help=(
            "Anki note type to use for imported cards (default: %(default)s). "
            "Use 'Basic (type in the answer)' if you explicitly want typed-answer cards."
        ),
    )
    romanized_group = parser.add_mutually_exclusive_group()
    romanized_group.add_argument(
        "--romanized",
        dest="include_romanized",
        action="store_true",
        help="Include romanized text when available (default behaviour).",
    )
    romanized_group.add_argument(
        "--no-romanized",
        dest="include_romanized",
        action="store_false",
        help="Skip romanized text in the generated cards.",
    )
    parser.set_defaults(include_romanized=False)
    return parser.parse_args()


def build_prompt(include_romanized: bool) -> str:
    romanized_line = (
        'Include a "romanized" key only when a romanization is available.\n'
        if include_romanized
        else "Do not include romanization keys in the output.\n"
    )
    return (
        "Read the attached PDF and extract vocabulary pairs.\n"
        'Return a JSON array where each item contains the keys "english" and "foreign" '
        "with string values.\n"
        f"{romanized_line}"
        "Respond with JSON only—no commentary, explanations, or additional fields."
    )


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


def normalize_json_payload(raw_output: str) -> str:
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_output = "\n".join(lines).strip()
    return raw_output


def parse_word_pairs(raw_output: str) -> List[Dict[str, Any]]:
    cleaned_output = normalize_json_payload(raw_output)
    if not cleaned_output:
        raise RuntimeError("Model returned an empty response; unable to extract vocabulary.")
    try:
        parsed = json.loads(cleaned_output)
    except json.JSONDecodeError as exc:
        snippet = cleaned_output[:200] + ("..." if len(cleaned_output) > 200 else "")
        raise RuntimeError(
            f"Failed to parse vocabulary JSON from model output. First 200 chars: {snippet}"
        ) from exc
    if not isinstance(parsed, list):
        raise RuntimeError("Model response was not a JSON array as requested.")
    return parsed


def build_note(deckname: str, front: str, back: str, model_name: str) -> Dict[str, Any]:
    return {
        "deckName": deckname,
        "modelName": model_name,
        "fields": {
            "Front": front,
            "Back": back,
        },
    }


def main():
    """
    Given a PDF file, this script converts it to a list of English word to foreign word pairs.
    The pairs are then added to an Anki deck as flashcards.
    The foreign word is the front of the card and the English word is the back.
    The deck name defaults to the PDF stem or can be provided via --deck.
    """
    args = parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    deckname = args.deck or args.pdf.stem
    print(f"Using deck name: {deckname}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Environment variable OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    # Getting the file ID
    print(f"Uploading PDF to OpenAI: {args.pdf}")
    file_id = create_file(client, args.pdf)

    prompt_text = build_prompt(args.include_romanized)

    response = client.responses.create(
        model=args.model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {
                    "type": "input_file",
                    "file_id": file_id,
                },
            ],
        }],
    )

    raw_output = get_response_text(response)
    word_pairs = parse_word_pairs(raw_output)

    if not args.dry_run:
        invoke('createDeck', deck=deckname)
        print(f"Deck '{deckname}' created. Preparing notes...")

    notes: Dict[str, Dict[str, Any]] = {}
    for vocab_pair in word_pairs:
        english = vocab_pair.get("english")
        foreign_word = vocab_pair.get("foreign")
        if not english or not foreign_word:
            continue
        english_clean = collapse_whitespace(english)
        foreign_clean = collapse_whitespace(foreign_word)
        if not english_clean or not foreign_clean:
            continue
        foreign_clean = strip_english_duplicate(foreign_clean, english_clean)
        romanized = vocab_pair.get("romanized") if args.include_romanized else None
        if romanized:
            romanized = collapse_whitespace(romanized)
            if not romanized:
                romanized = None
        if romanized:
            foreign_display = f"{foreign_clean} ({romanized})"
        else:
            foreign_display = foreign_clean
        if foreign_clean in notes:
            print(f"Skipping duplicate entry for: {foreign_clean}")
            continue
        notes[foreign_clean] = build_note(
            deckname,
            foreign_display,
            english_clean,
            args.note_model,
        )

    if args.dry_run:
        print(f"{len(notes)} notes ready. Dry-run mode; no changes made.")
        sample = list(notes.values())[:10]
        if sample:
            print("Sample notes:")
            for note in sample:
                front = note["fields"]["Front"]
                back = note["fields"]["Back"]
                print(f"- {front} — {back}")
        summary = {
            "ok": True,
            "dry_run": True,
            "deck": deckname,
            "notes_prepared": len(notes),
            "notes_added": 0,
        }
        print(f"SUMMARY: {json.dumps(summary, ensure_ascii=False)}")
        return

    invoke("addNotes", notes=list(notes.values()))
    print(f"Added {len(notes)} notes to deck '{deckname}'.")
    summary = {
        "ok": True,
        "dry_run": False,
        "deck": deckname,
        "notes_prepared": len(notes),
        "notes_added": len(notes),
    }
    print(f"SUMMARY: {json.dumps(summary, ensure_ascii=False)}")
    try:
        pdf_archive_dir = Path.cwd() / ".local_memory" / "pdfs"
        pdf_archive_dir.mkdir(parents=True, exist_ok=True)
        destination = pdf_archive_dir / args.pdf.name
        if destination.exists():
            print(f"PDF already exists at {destination}; skipping move.")
        else:
            shutil.move(str(args.pdf), destination)
            print(f"Moved processed PDF to {destination}.")
    except Exception as exc:
        print(f"Warning: Failed to archive PDF: {exc}")

if __name__=="__main__":
    main()
