"""Business-address normalization and lightweight component extraction."""

from __future__ import annotations

import re

from .tokenization import tokenize

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "ct": "court",
    "apt": "apartment",
    "ste": "suite",
    "rm": "room",
    "no": "number",
}

POSTAL_RE = re.compile(r"\b\d{4,8}\b")


def normalize_address(text: str) -> str:
    tokens = tokenize(text)
    expanded = [ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens]
    return " ".join(expanded)


def address_tokens(text: str) -> list[str]:
    return normalize_address(text).split()


def postal_tokens(text: str) -> list[str]:
    normalized = normalize_address(text)
    return POSTAL_RE.findall(normalized)


def numeric_tokens(text: str) -> set[str]:
    return {token for token in address_tokens(text) if token.isdigit()}
