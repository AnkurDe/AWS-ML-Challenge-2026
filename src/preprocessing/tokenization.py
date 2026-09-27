"""Shared text tokenization helpers."""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

TOKEN_RE = re.compile(r"[a-z0-9]+")


def ascii_normalize(text: str) -> str:
    """Normalize unicode and transliterate where possible."""
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    """Return alphanumeric word tokens."""
    return TOKEN_RE.findall(ascii_normalize(text))


def compact_text(text: str) -> str:
    """Normalize text into a single whitespace-separated representation."""
    return " ".join(tokenize(text))
