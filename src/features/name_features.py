"""Name similarity features."""

from __future__ import annotations

from rapidfuzz import fuzz


def name_features(left: str, right: str) -> dict[str, float]:
    left = left or ""
    right = right or ""
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    intersection = left_tokens & right_tokens
    return {
        "name_ratio": fuzz.ratio(left, right) / 100.0,
        "name_token_set": fuzz.token_set_ratio(left, right) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(left, right) / 100.0,
        "name_jaccard": (len(intersection) / len(union)) if union else 1.0,
        "name_exact": float(left == right and bool(left)),
        "name_length_ratio": _length_ratio(left, right),
    }


def _length_ratio(left: str, right: str) -> float:
    a, b = len(left), len(right)
    if max(a, b) == 0:
        return 1.0
    return min(a, b) / max(a, b)
