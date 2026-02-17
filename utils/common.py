from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "media"
IMAGE_DIR = MEDIA_DIR / "images"
HTML_TAG_RE = re.compile(r"<[^>]+>")
IMG_TAG_RE = re.compile(r"<img[^>]*?>", re.IGNORECASE)
IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\'>]+)["\']', re.IGNORECASE)
SOUND_TAG_RE = re.compile(r"\[sound:[^\]]+\]")
NBSP_RE = re.compile(r"&nbsp;?", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", (text or "")).strip()


def strip_english_duplicate(foreign_text: str, english_text: str) -> str:
    """
    Remove a duplicated English gloss accidentally merged into the foreign side.

    Handles common patterns like:
    - "성격 personality"
    - "personality 성격"
    - "성격 (personality)"
    - "성격 - personality"
    """
    foreign = collapse_whitespace(foreign_text)
    english = collapse_whitespace(english_text)
    if not foreign or not english:
        return foreign

    escaped = re.escape(english)
    cleaned = foreign
    patterns = [
        rf"\s*\({escaped}\)\s*$",
        rf"\s*[-–—:|/]\s*{escaped}\s*$",
        rf"\s+{escaped}\s*$",
        rf"^{escaped}\s*[-–—:|/]\s*",
        rf"^{escaped}\s+",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = collapse_whitespace(cleaned.strip(" -–—:|/()"))

    if not cleaned:
        return foreign
    if cleaned.casefold() == english.casefold():
        return foreign
    return cleaned
