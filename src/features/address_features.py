"""Address similarity features."""

from __future__ import annotations

from rapidfuzz import fuzz

from ..preprocessing import numeric_tokens, postal_tokens


def address_features(left: str, right: str) -> dict[str, float]:
    left = left or ""
    right = right or ""
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    intersection = left_tokens & right_tokens
    left_numbers = numeric_tokens(left)
    right_numbers = numeric_tokens(right)
    left_postal = set(postal_tokens(left))
    right_postal = set(postal_tokens(right))
    return {
        "address_ratio": fuzz.ratio(left, right) / 100.0,
        "address_token_set": fuzz.token_set_ratio(left, right) / 100.0,
        "address_token_sort": fuzz.token_sort_ratio(left, right) / 100.0,
        "address_jaccard": (len(intersection) / len(union)) if union else 1.0,
        "address_exact": float(left == right and bool(left)),
        "number_overlap": float(bool(left_numbers & right_numbers)),
        "postal_overlap": float(bool(left_postal & right_postal)),
        "address_length_ratio": _length_ratio(left, right),
    }


def _length_ratio(left: str, right: str) -> float:
    a, b = len(left), len(right)
    if max(a, b) == 0:
        return 1.0
    return min(a, b) / max(a, b)
