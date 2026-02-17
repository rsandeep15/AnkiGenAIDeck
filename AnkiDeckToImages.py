import argparse
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import re
from pathlib import Path
from typing import Any, List, Tuple

from openai import OpenAI

from utils.anki_connect import invoke
from config import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_WORKERS,
    GATING_PROMPT_ID,
    GATING_PROMPT_VERSION,
)
from utils.common import (
    BASE_DIR,
    IMAGE_DIR,
    HTML_TAG_RE,
    IMG_TAG_RE,
    NBSP_RE,
)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_TERMS = {
    "happy",
    "angry",
    "sad",
    "bored",
    "worry",
    "worried",
    "tired",
    "glad",
    "satisfied",
    "uncomfortable",
    "nervous",
    "scary",
    "scared",
    "excited",
    "joyful",
    "comfortable",
    "shy",
    "lonely",
    "surprised",
    "relaxed",
    "anxious",
    "calm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate illustrative images for Anki notes in a specified deck."
    )
    parser.add_argument("deck", help="Name of the Anki deck to process.")
    parser.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help="Image generation model to use (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Generate a memory aid illustration for this Anki flashcard concept: {text}. "
            "Do not include any words or letters. Favor stylized anime/cartoon aesthetics, not photorealism."
        ),
        help="Template used for image generation; {text} is replaced with the back of the card.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("ANKI_IMAGE_WORKERS", str(DEFAULT_IMAGE_WORKERS))),
        help="Maximum number of concurrent generations (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-gating",
        action="store_true",
        help="Generate images for every eligible card without the gating check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List eligible cards and print a summary without writing to Anki.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N candidates after optional shuffling (default: all).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle candidate order before applying --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --shuffle for reproducible batches (default: %(default)s).",
    )
    return parser.parse_args()


def load_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Environment variable OPENAI_API_KEY is not set.")
    return api_key


def get_candidate_cards(deckname: str) -> List[Tuple[int, str, str]]:
    # Quote deck name to handle spaces/special characters like '&'
    cards = invoke("findNotes", query=f'deck:"{deckname}"')
    if not cards:
        return []
    notes_info = invoke("notesInfo", notes=cards)
    candidates: List[Tuple[int, str, str]] = []
    for card_id, note in zip(cards, notes_info):
        front_text = note["fields"]["Front"]["value"]
        back_text = note["fields"]["Back"]["value"]
        candidates.append((card_id, front_text, back_text))
    return candidates


def select_candidates(
    candidates: List[Tuple[int, str, str]],
    *,
    limit: int,
    shuffle: bool,
    seed: int,
) -> List[Tuple[int, str, str]]:
    selected = list(candidates)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if limit > 0:
        selected = selected[:limit]
    return selected


def sanitize_text(text: str) -> str:
    """Strip HTML tags and collapse whitespace for cleaner prompts."""
    without_nbsp = NBSP_RE.sub(" ", text)
    without_tags = HTML_TAG_RE.sub(" ", without_nbsp)
    return " ".join(without_tags.split())


def strip_image_tags(text: str) -> str:
    """Remove any <img> tags so we can replace or delete them cleanly."""
    return IMG_TAG_RE.sub("", text)


def build_image_prompt(template: str, concept: str) -> str:
    return template.format(text=concept)


def generate_image(
    client: OpenAI,
    prompt: str,
    filename: str,
    *,
    model: str,
) -> Path:
    result = client.images.generate(
        model=model,
        prompt=prompt,
    )
    image_base64 = result.data[0].b64_json
    target_path = IMAGE_DIR / filename
    with open(target_path, "wb") as handle:
        handle.write(base64.b64decode(image_base64))
    return target_path.resolve()


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


def parse_gating_decision(raw: str) -> bool:
    decision = (raw or "").strip().lower()
    if not decision:
        return False
    if decision in {"true", "yes", "1"}:
        return True
    if decision in {"false", "no", "0"}:
        return False
    if re.match(r"^true\b", decision):
        return True
    if re.match(r"^false\b", decision):
        return False
    try:
        payload = json.loads(decision)
        if isinstance(payload, bool):
            return payload
        if isinstance(payload, dict):
            for key in ("allow", "should_generate", "generate"):
                value = payload.get(key)
                if isinstance(value, bool):
                    return value
    except json.JSONDecodeError:
        pass
    return False


def emotion_state_prior(front_text: str, back_text: str) -> bool:
    text = f"{front_text} {back_text}".lower()
    if re.search(r"\bto\s+(be|feel)\b", text):
        return True
    return any(term in text for term in EMOTION_TERMS)


def should_generate_image(
    client: OpenAI,
    front_text: str,
    back_text: str,
) -> bool:
    if emotion_state_prior(front_text, back_text):
        return True

    prompt_payload = {
        "id": GATING_PROMPT_ID,
        "version": GATING_PROMPT_VERSION,
        "variables": {
            "front": front_text,
            "back": back_text,
        },
    }
    response = client.responses.create(
        prompt=prompt_payload,
    )
    decision = get_response_text(response)
    return parse_gating_decision(decision)


def process_card(
    card: Tuple[int, str, str],
    api_key: str,
    image_model: str,
    prompt_template: str,
    skip_gating: bool,
    dry_run: bool,
) -> Tuple[str, str, Any]:
    card_id, front_text, back_text = card
    local_client = OpenAI(api_key=api_key)
    front_without_images = strip_image_tags(front_text)
    back_without_images = strip_image_tags(back_text)
    cleaned_back = sanitize_text(back_without_images)
    if not cleaned_back:
        return ("skip", back_without_images, "No descriptive text after cleaning.")

    try:
        should_generate = True
        if not skip_gating:
            cleaned_front = sanitize_text(front_without_images)
            should_generate = should_generate_image(
                local_client,
                cleaned_front or front_without_images,
                cleaned_back,
            )
            if not should_generate:
                if dry_run:
                    return ("would_skip", back_without_images, "Gating model returned false.")
                if (
                    front_without_images != front_text
                    or back_without_images != back_text
                ):
                    invoke(
                        "updateNoteFields",
                        note={
                            "id": card_id,
                            "fields": {
                                "Front": front_without_images,
                                "Back": back_without_images,
                            },
                        },
                    )
                    return (
                        "skip",
                        back_without_images,
                        "Gating model returned false; existing image removed.",
                    )
                return ("skip", back_without_images, "Gating model returned false.")
        if dry_run:
            if should_generate:
                return ("would_add", back_without_images, None)
            return ("would_skip", back_without_images, "Gating model returned false.")

        filename = f"{card_id}.png"
        prompt = build_image_prompt(prompt_template, cleaned_back)
        file_path = generate_image(local_client, prompt, filename, model=image_model)
        invoke(
            "updateNoteFields",
            note={
                "id": card_id,
                "fields": {
                    "Front": front_without_images,
                    "Back": back_without_images,
                },
                "picture": [
                    {
                        "filename": filename,
                        "fields": ["Front"],
                        "path": file_path.as_posix(),
                    }
                ],
            },
        )
        return ("added", back_text, None)
    except Exception as exc:
        return ("error", back_text, exc)


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    print(f"Fetching notes for deck: {args.deck}")
    candidates = get_candidate_cards(args.deck)
    if not candidates:
        print(f"No cards eligible for image generation in deck '{args.deck}'.")
        return
    candidates = select_candidates(
        candidates,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if args.shuffle:
        print(f"Shuffled candidates with seed={args.seed}.")
    if args.limit > 0:
        print(f"Processing a limited batch of {len(candidates)} candidate(s).")

    if args.dry_run:
        print("Dry-run mode; evaluating gating decisions without generating images.")

    worker_limit = max(1, args.workers)
    max_workers = max(1, min(worker_limit, len(candidates)))
    prompt_template = args.prompt.strip()
    print(
        f"Generating images with up to {max_workers} worker(s) using image model {args.image_model} "
        f"and {'skipping' if args.skip_gating else 'using prompt-configured'} gating."
    )

    added = skipped = failed = dry_run_count = 0
    total = len(candidates)
    processed = 0
    print(f"PROGRESS 0/{total}", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_card,
                card,
                api_key,
                args.image_model,
                prompt_template,
                args.skip_gating,
                args.dry_run,
            )
            for card in candidates
        ]
        for future in as_completed(futures):
            status, back_text, error = future.result()
            if status == "added":
                print(f"Adding image for: {back_text}")
                added += 1
            elif status == "skip":
                print(f"Skipping image for: {back_text} ({error})")
                skipped += 1
            elif status == "dry_run":
                dry_run_count += 1
            elif status == "would_add":
                print(f"Would add image for: {back_text}")
                added += 1
            elif status == "would_skip":
                print(f"Would skip image for: {back_text} ({error})")
                skipped += 1
            else:
                print(f"Failed image for: {back_text} ({error})")
                failed += 1
            processed += 1
            print(f"PROGRESS {processed}/{total}", flush=True)

    print(f"Completed image generation: {added} added, {skipped} skipped, {failed} failed.")
    summary = {
        "ok": failed == 0,
        "dry_run": args.dry_run,
        "deck": args.deck,
        "candidates": len(candidates),
        "added": added,
        "skipped": skipped,
        "failed": failed,
        "dry_run_count": dry_run_count,
    }
    print(f"SUMMARY: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
