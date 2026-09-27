"""Multi-pass candidate generation for Source 1 -> Source 2/3."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz

from ..preprocessing import normalize_address, normalize_name_core
from .address_blocking import AddressBlockIndex
from .name_blocking import NameBlockIndex


@dataclass(frozen=True)
class BlockingConfig:
    max_candidates_per_entity: int = 80
    max_name_block_frequency: int = 250
    max_name_ngram_frequency: int = 500
    max_address_block_frequency: int = 250


class CandidateGenerator:
    """Generate a small, high-recall candidate set per Source 1 record.

    All retrieval operations use inverted indexes. No per-entity full-country
    scan is performed, which keeps the blocking stage scalable.
    """

    def __init__(self, config: BlockingConfig | None = None) -> None:
        self.config = config or BlockingConfig()
        self._targets: pd.DataFrame | None = None
        self._name_index = NameBlockIndex(
            self.config.max_name_block_frequency,
            self.config.max_name_ngram_frequency,
        )
        self._address_index = AddressBlockIndex(self.config.max_address_block_frequency)
        self._target_by_id: dict[str, dict] = {}

    def fit(self, source2: pd.DataFrame, source3: pd.DataFrame) -> "CandidateGenerator":
        targets = pd.concat([source2, source3], ignore_index=True)
        self._targets = targets
        self._name_index.fit(targets)
        self._address_index.fit(targets)
        self._target_by_id = {
            str(row.entity_id): row._asdict() for row in targets.itertuples(index=False)
        }
        return self

    def _rank(self, source_row, candidates: set[str]) -> list[str]:
        if not candidates:
            return []
        query_name = normalize_name_core(source_row.business_name)
        query_address = normalize_address(source_row.business_address)
        ranked: list[tuple[float, str]] = []
        for entity_id in candidates:
            target = self._target_by_id[entity_id]
            name_score = fuzz.token_set_ratio(query_name, target["name_norm_core"])
            address_score = fuzz.token_set_ratio(query_address, target["address_norm"])
            score = 0.65 * name_score + 0.35 * address_score
            ranked.append((score, entity_id))
        ranked.sort(reverse=True)
        return [entity_id for _, entity_id in ranked[: self.config.max_candidates_per_entity]]

    def generate_for_row(self, source_row) -> list[str]:
        candidates = self._name_index.lookup(source_row.country_norm, source_row.business_name)
        candidates.update(self._address_index.lookup(source_row.country_norm, source_row.business_address))
        return self._rank(source_row, candidates)

    def generate(self, source1: pd.DataFrame) -> dict[str, list[str]]:
        if self._targets is None:
            raise RuntimeError("CandidateGenerator.fit must be called first.")
        return {
            str(row.entity_id): self.generate_for_row(row)
            for row in source1.itertuples(index=False)
        }
