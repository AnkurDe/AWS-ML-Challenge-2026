"""Address-oriented blocking index."""

from __future__ import annotations

from collections import defaultdict, Counter

import pandas as pd

from src.preprocessing import address_tokens, postal_tokens


class AddressBlockIndex:
    """Indexes target records using postal, numeric and rare address tokens."""

    def __init__(self, max_token_frequency: int = 250) -> None:
        self.max_token_frequency = max_token_frequency
        self.postal_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.rare_token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.numeric_index: dict[tuple[str, str], set[str]] = defaultdict(set)

    def fit(self, frame: pd.DataFrame) -> "AddressBlockIndex":
        frequency: Counter[tuple[str, str]] = Counter()
        # Count first, then index on a second pass. Retaining token sets for all
        # targets together causes a large peak allocation.
        for row in frame.itertuples(index=False):
            country = str(row.country_norm)
            tokens = {token for token in address_tokens(row.address_norm) if len(token) >= 4}
            for token in tokens:
                frequency[(country, token)] += 1

        for row in frame.itertuples(index=False):
            country = str(row.country_norm)
            entity_id = str(row.entity_id)
            tokens = {token for token in address_tokens(row.address_norm) if len(token) >= 4}
            postal = set(postal_tokens(row.address_norm))
            numbers = {
                token for token in address_tokens(row.address_norm)
                if token.isdigit() and 2 <= len(token) <= 8
            }
            for token in postal:
                self.postal_index[(country, token)].add(entity_id)
            for number in numbers:
                self.numeric_index[(country, number)].add(entity_id)
            for token in tokens:
                if frequency[(country, token)] <= self.max_token_frequency:
                    self.rare_token_index[(country, token)].add(entity_id)
        return self

    def lookup(self, country: str, business_address: str) -> set[str]:
        country = str(country)
        candidates: set[str] = set()
        for token in postal_tokens(business_address):
            candidates.update(self.postal_index.get((country, token), ()))
        for token in address_tokens(business_address):
            if token.isdigit() and 2 <= len(token) <= 8:
                candidates.update(self.numeric_index.get((country, token), ()))
            elif len(token) >= 4:
                candidates.update(self.rare_token_index.get((country, token), ()))
        return candidates
