"""Name-oriented blocking index."""

from __future__ import annotations

from collections import defaultdict, Counter

import pandas as pd

from src.preprocessing import name_first_token, name_prefix, normalize_name_core


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = f" {text.strip()} "
    if len(text) < n:
        return {text} if text.strip() else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


class NameBlockIndex:
    """Indexes target records using exact keys, rare tokens and rare character n-grams."""

    def __init__(self, max_token_frequency: int = 250, max_ngram_frequency: int = 500) -> None:
        self.max_token_frequency = max_token_frequency
        self.max_ngram_frequency = max_ngram_frequency
        self.prefix_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.first_token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.rare_token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.ngram_index: dict[tuple[str, str], set[str]] = defaultdict(set)

    def fit(self, frame: pd.DataFrame) -> "NameBlockIndex":
        token_frequency: Counter[tuple[str, str]] = Counter()
        ngram_frequency: Counter[tuple[str, str]] = Counter()
        # First pass counts frequencies. Avoid retaining every row's token and
        # n-gram sets: at this dataset size that intermediate is enormous.
        for row in frame.itertuples(index=False):
            country = str(row.country_norm)
            tokens = set(row.name_norm_core.split())
            ngrams = char_ngrams(row.name_norm_core)
            for token in tokens:
                token_frequency[(country, token)] += 1
            for ngram in ngrams:
                ngram_frequency[(country, ngram)] += 1

        # Second pass builds the inverted indexes with one row's tokens live at a time.
        for row in frame.itertuples(index=False):
            country = str(row.country_norm)
            entity_id = str(row.entity_id)
            core = row.name_norm_core
            prefix = name_prefix(core)
            first = name_first_token(core)
            tokens = set(core.split())
            ngrams = char_ngrams(core)
            if prefix:
                self.prefix_index[(country, prefix)].add(entity_id)
            if first:
                self.first_token_index[(country, first)].add(entity_id)
            for token in tokens:
                if token_frequency[(country, token)] <= self.max_token_frequency:
                    self.rare_token_index[(country, token)].add(entity_id)
            for ngram in ngrams:
                if ngram_frequency[(country, ngram)] <= self.max_ngram_frequency:
                    self.ngram_index[(country, ngram)].add(entity_id)
        return self

    def lookup(self, country: str, business_name: str) -> set[str]:
        country = str(country)
        candidates: set[str] = set()
        prefix = name_prefix(business_name)
        first = name_first_token(business_name)
        if prefix:
            candidates.update(self.prefix_index.get((country, prefix), ()))
        if first:
            candidates.update(self.first_token_index.get((country, first), ()))

        core = normalize_name_core(business_name)
        for token in set(core.split()):
            candidates.update(self.rare_token_index.get((country, token), ()))
        for ngram in char_ngrams(core):
            candidates.update(self.ngram_index.get((country, ngram), ()))
        return candidates
