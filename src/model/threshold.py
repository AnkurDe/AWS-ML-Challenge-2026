"""Threshold selection and competition metric."""

from __future__ import annotations

from collections import defaultdict


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    denominator = beta * beta * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta * beta) * precision * recall / denominator


def entity_f05(predicted: set[str], actual: set[str]) -> float:
    if not predicted and not actual:
        return 1.0
    if not predicted:
        return 0.0
    if not actual:
        return 0.0
    tp = len(predicted & actual)
    precision = tp / len(predicted)
    recall = tp / len(actual)
    return fbeta(precision, recall, beta=0.5)


def macro_f05(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    scores = [entity_f05(predictions.get(entity_id, set()), actual) for entity_id, actual in truth.items()]
    return sum(scores) / len(scores) if scores else 0.0


def tune_threshold(
    probabilities: list[float],
    keys: list[tuple[str, str]],
    truth: dict[str, set[str]],
    minimum: float = 0.30,
    maximum: float = 0.98,
    steps: int = 35,
) -> tuple[float, float]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (s1_id, candidate_id), probability in zip(keys, probabilities):
        grouped[s1_id].append((candidate_id, probability))

    best_threshold = minimum
    best_score = -1.0
    thresholds = [minimum + i * (maximum - minimum) / max(steps - 1, 1) for i in range(steps)]
    for threshold in thresholds:
        predictions = {
            s1_id: {candidate_id for candidate_id, probability in rows if probability >= threshold}
            for s1_id, rows in grouped.items()
        }
        score = macro_f05(predictions, truth)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold, best_score
