import argparse
import json
import os
from pathlib import Path
import sys
import shutil
import urllib.error
import urllib.request
from typing import Any, Dict, List

from openai import OpenAI

ANKI_CONNECT_URL = "http://127.0.0.1:8765"

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


def invoke(action: str, **params: Any) -> Any:
    request_json = json.dumps(request(action, **params)).encode("utf-8")
    req = urllib.request.Request(ANKI_CONNECT_URL, request_json)
    try:
        with urllib.request.urlopen(req) as response_handle:
            response = json.load(response_handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach AnkiConnect at {ANKI_CONNECT_URL}: {exc}") from exc
    if len(response) != 2:
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        raise Exception('response is missing required error field')
    if 'result' not in response:
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']


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
        default="gpt-4.1-mini",
        help=(
            "OpenAI model used to extract vocabulary (default: %(default)s). "
            "Try faster tiers like 'gpt-4o-mini' or higher-accuracy models like 'gpt-4.1'."
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


def build_note(deckname: str, front: str, back: str) -> Dict[str, Any]:
    return {
        "deckName": deckname,
        "modelName": "Basic (type in the answer)",
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

    invoke('createDeck', deck=deckname)
    print(f"Deck '{deckname}' created. Preparing notes...")

    notes: Dict[str, Dict[str, Any]] = {}
    for vocab_pair in word_pairs:
        english = vocab_pair.get("english")
        foreign_word = vocab_pair.get("foreign")
        if not english or not foreign_word:
            continue
        english_clean = english.strip()
        foreign_clean = foreign_word.strip()
        if not english_clean or not foreign_clean:
            continue
        romanized = vocab_pair.get("romanized") if args.include_romanized else None
        if romanized:
            romanized = romanized.strip()
            if not romanized:
                romanized = None
        if romanized:
            foreign_display = f"{foreign_clean} ({romanized})"
        else:
            foreign_display = foreign_clean
        if foreign_clean in notes:
            print(f"Skipping duplicate entry for: {foreign_clean}")
            continue
        notes[foreign_clean] = build_note(deckname, foreign_display, english_clean)

    invoke("addNotes", notes=list(notes.values()))
    print(f"Added {len(notes)} notes to deck '{deckname}'.")
    try:
        pdf_archive_dir = Path.cwd() / "pdfs"
        pdf_archive_dir.mkdir(exist_ok=True)
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
