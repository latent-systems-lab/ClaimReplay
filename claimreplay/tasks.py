"""Evaluation tasks map canonical labels to explicitly named targets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import SupportLevel


class UnassessedHandling(str, Enum):
    EXCLUDE_FROM_METRICS = "EXCLUDE_FROM_METRICS"
    ERROR = "ERROR"


@dataclass(frozen=True)
class EvaluationTask:
    name: str
    metric_prefix: str
    positive_labels: frozenset[SupportLevel]
    eligible_labels: frozenset[SupportLevel] = frozenset({
        SupportLevel.FULL_SUPPORT,
        SupportLevel.PARTIAL_SUPPORT,
        SupportLevel.NO_SUPPORT,
    })
    unassessed_handling: UnassessedHandling = UnassessedHandling.EXCLUDE_FROM_METRICS

    def __post_init__(self) -> None:
        if not self.positive_labels:
            raise ValueError("EvaluationTask requires at least one positive label")
        if SupportLevel.UNASSESSED in self.positive_labels:
            raise ValueError("UNASSESSED cannot be a positive evaluation label")
        if SupportLevel.UNASSESSED in self.eligible_labels:
            raise ValueError("UNASSESSED cannot be an eligible evaluation label")
        if not self.positive_labels <= self.eligible_labels:
            raise ValueError("Positive labels must be a subset of eligible labels")


def unsupported_detection() -> EvaluationTask:
    return EvaluationTask(
        name="unsupported_detection",
        metric_prefix="Unsupported",
        positive_labels=frozenset({SupportLevel.NO_SUPPORT}),
    )


def review_need_detection() -> EvaluationTask:
    return EvaluationTask(
        name="review_need_detection",
        metric_prefix="ReviewNeeded",
        positive_labels=frozenset({SupportLevel.PARTIAL_SUPPORT, SupportLevel.NO_SUPPORT}),
    )
