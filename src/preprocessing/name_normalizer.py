"""Business-name normalization."""

from __future__ import annotations

import re

from .tokenization import compact_text, tokenize

ABBREVIATIONS = {
    "pvt": "private",
    "pvtltd": "private limited",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "llc": "limited liability company",
    "intl": "international",
}

LEGAL_SUFFIXES = {
    "private",
    "limited",
    "corporation",
    "company",
    "incorporated",
    "llc",
    "plc",
    "lp",
    "llp",
}


def _expand(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        replacement = ABBREVIATIONS.get(token)
        if replacement:
            expanded.extend(replacement.split())
        else:
            expanded.append(token)
    return expanded


def normalize_name(text: str) -> str:
    """Normalize a business name while retaining useful semantic tokens."""
    return " ".join(_expand(tokenize(text)))


def normalize_name_core(text: str) -> str:
    """Normalize a business name and remove trailing legal-form tokens."""
    tokens = _expand(tokenize(text))
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def name_prefix(text: str, width: int = 4) -> str:
    core = normalize_name_core(text).replace(" ", "")
    return core[:width]


def name_first_token(text: str) -> str:
    tokens = tokenize(normalize_name_core(text))
    return tokens[0] if tokens else ""
