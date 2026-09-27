"""Official entity-level macro F0.5 evaluation."""

from __future__ import annotations


def entity_f05(truth: set[str], predicted: set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(truth & predicted)
    if not tp:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_entity_f05(
    truths: dict[str, set[str]], predictions: dict[str, set[str]]
) -> float:
    if not truths:
        return 0.0
    return sum(
        entity_f05(truth, predictions.get(s1_id, set()))
        for s1_id, truth in truths.items()
    ) / len(truths)

