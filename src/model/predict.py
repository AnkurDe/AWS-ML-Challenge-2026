"""Prediction helpers for final inference."""

from __future__ import annotations

from collections import defaultdict


def build_predictions(keys, probabilities, threshold: float, source1_ids):
    predictions = defaultdict(set)
    for (s1_id, candidate_id), probability in zip(keys, probabilities):
        if probability >= threshold:
            predictions[s1_id].add(candidate_id)
    return {s1_id: predictions.get(s1_id, set()) for s1_id in source1_ids}
