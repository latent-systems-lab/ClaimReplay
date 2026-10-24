"""Agreement and matched-label sensitivity metrics for surrogate judges."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from statistics import fmean
from typing import Iterable, Mapping, Sequence


SUPPORT_LABELS = ("FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT")


def confusion_matrix(
    human: Sequence[str], predicted: Sequence[str], *, labels: Sequence[str] = SUPPORT_LABELS
) -> dict[str, dict[str, int]]:
    if len(human) != len(predicted):
        raise ValueError("Human and predicted sequences must have equal length")
    allowed = set(labels)
    if any(value not in allowed for value in (*human, *predicted)):
        raise ValueError("Agreement inputs contain an unknown or abstaining label")
    matrix = {truth: {guess: 0 for guess in labels} for truth in labels}
    for truth, guess in zip(human, predicted):
        matrix[truth][guess] += 1
    return matrix


def agreement_metrics(
    human: Sequence[str], predicted: Sequence[str], *, labels: Sequence[str] = SUPPORT_LABELS
) -> dict:
    matrix = confusion_matrix(human, predicted, labels=labels)
    total = len(human)
    if total == 0:
        raise ValueError("Agreement metrics require at least one matched judgment")
    by_class = {}
    recalls = []
    f1s = []
    for label in labels:
        tp = matrix[label][label]
        truth_total = sum(matrix[label].values())
        predicted_total = sum(matrix[truth][label] for truth in labels)
        recall = tp / truth_total if truth_total else None
        precision = tp / predicted_total if predicted_total else None
        # A represented class with zero true positives contributes zero F1.
        # Only a class absent from both reference and prediction is undefined.
        f1_denominator = truth_total + predicted_total
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        by_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": truth_total}
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)
    agreement = sum(matrix[label][label] for label in labels) / total
    human_counts = Counter(human)
    predicted_counts = Counter(predicted)
    chance = sum(human_counts[label] * predicted_counts[label] for label in labels) / (total * total)
    kappa = None if math.isclose(chance, 1.0) else (agreement - chance) / (1.0 - chance)
    return {
        "count": total,
        "labels": list(labels),
        "confusion_matrix": matrix,
        "per_class": by_class,
        "exact_agreement": agreement,
        "balanced_accuracy": fmean(recalls),
        "macro_f1": fmean(f1s) if f1s else None,
        "cohen_kappa": kappa,
    }


def review_needed(label: str) -> bool:
    if label not in SUPPORT_LABELS:
        raise ValueError(f"Unsupported label {label!r}")
    return label in {"PARTIAL_SUPPORT", "NO_SUPPORT"}


def binary_agreement_metrics(human: Sequence[str], predicted: Sequence[str]) -> dict:
    left = ["REVIEW_NEEDED" if review_needed(value) else "FULL_SUPPORT" for value in human]
    right = ["REVIEW_NEEDED" if review_needed(value) else "FULL_SUPPORT" for value in predicted]
    return agreement_metrics(left, right, labels=("FULL_SUPPORT", "REVIEW_NEEDED"))


def topic_bootstrap_interval(
    values_by_topic: Mapping[str, float], *, seed: int = 47, resamples: int = 10000
) -> dict:
    values = {str(key): float(value) for key, value in values_by_topic.items()}
    if len(values) < 2 or resamples < 1:
        raise ValueError("At least two topics and one resample are required")
    topics = sorted(values)
    rng = random.Random(seed)
    draws = sorted(fmean(values[rng.choice(topics)] for _ in topics) for _ in range(resamples))

    def quantile(probability: float) -> float:
        position = probability * (len(draws) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return draws[lower]
        weight = position - lower
        return draws[lower] * (1 - weight) + draws[upper] * weight

    return {
        "estimate": fmean(values.values()),
        "confidence_interval": [quantile(.025), quantile(.975)],
        "topic_count": len(topics),
        "resamples": resamples,
        "seed": seed,
    }


def matched_fixed_selection_recall(
    units: Iterable[Mapping[str, str]],
    selected_ids: Iterable[str],
    *,
    label_field: str,
) -> dict:
    """Evaluate one label plane without replacing unmatched inspection slots."""
    rows = list(units)
    selected = set(selected_ids)
    known_ids = {str(row["unit_id"]) for row in rows}
    if not selected <= known_ids:
        raise ValueError("Selected IDs must belong to the original response")
    matched = [row for row in rows if row.get("human_label") in SUPPORT_LABELS
               and row.get("llm_label") in SUPPORT_LABELS]
    positives = {str(row["unit_id"]) for row in matched if review_needed(str(row[label_field]))}
    selected_matched = {str(row["unit_id"]) for row in matched} & selected
    found = len(positives & selected)
    return {
        "positive_count": len(positives),
        "found": found,
        "recall": None if not positives else found / len(positives),
        "selected_total": len(selected),
        "selected_matched": len(selected_matched),
        "matched_selection_coverage": None if not selected else len(selected_matched) / len(selected),
    }
