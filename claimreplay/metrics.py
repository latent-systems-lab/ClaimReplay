"""Budget-aware metrics for an explicitly defined evaluation task."""

from __future__ import annotations

from dataclasses import dataclass

from .models import FrozenResponse, SupportLevel
from .tasks import EvaluationTask


@dataclass(frozen=True)
class ReplayMetrics:
    task_name: str
    metric_prefix: str
    total_unit_count: int
    positive_count: int
    eligible_unit_count: int
    unassessed_unit_count: int
    selected_count: int
    selected_eligible_count: int
    found: int
    recall: float | None
    precision: float | None
    random_expected_recall: float | None
    assessment_coverage: float
    selected_assessment_coverage: float | None
    clean_response: bool
    fully_assessed: bool

    def as_dict(self) -> dict[str, int | float | bool | None | str]:
        prefix = self.metric_prefix
        return {
            "task": self.task_name,
            "TotalUnitCount": self.total_unit_count,
            f"{prefix}Count": self.positive_count,
            "EligibleUnitCount": self.eligible_unit_count,
            "UnassessedUnitCount": self.unassessed_unit_count,
            "SelectedCount": self.selected_count,
            "SelectedEligibleCount": self.selected_eligible_count,
            f"{prefix}Found": self.found,
            f"{prefix}Recall": self.recall,
            f"{prefix}Precision": self.precision,
            "RandomExpectedRecall": self.random_expected_recall,
            "AssessmentCoverage": self.assessment_coverage,
            "SelectedAssessmentCoverage": self.selected_assessment_coverage,
            "CleanResponse": self.clean_response,
            "FullyAssessed": self.fully_assessed,
        }


def compute_metrics(
    response: FrozenResponse, selected_unit_ids: tuple[str, ...], task: EvaluationTask
) -> ReplayMetrics:
    known_ids = {unit.unit_id for unit in response.units}
    if len(selected_unit_ids) != len(set(selected_unit_ids)):
        raise ValueError("Selected unit IDs must be unique")
    if not set(selected_unit_ids) <= known_ids:
        raise ValueError("Selected unit IDs must belong to the frozen response")
    labels = response.assessment_map()
    unassessed = {
        unit_id for unit_id, assessment in labels.items()
        if assessment.support is SupportLevel.UNASSESSED
    }
    if unassessed and task.unassessed_handling.value == "ERROR":
        raise ValueError(
            f"Task {task.name!r} forbids {len(unassessed)} UNASSESSED units"
        )
    eligible = {
        unit_id for unit_id, assessment in labels.items()
        if assessment.support in task.eligible_labels
    }
    positives = {
        unit_id
        for unit_id, assessment in labels.items()
        if assessment.support in task.positive_labels
    }
    found = len(positives & set(selected_unit_ids))
    positive_count = len(positives)
    selected_count = len(selected_unit_ids)
    selected_eligible_count = len(set(selected_unit_ids) & eligible)
    fully_assessed = len(unassessed) == 0
    clean = positive_count == 0 and fully_assessed
    return ReplayMetrics(
        task_name=task.name,
        metric_prefix=task.metric_prefix,
        total_unit_count=len(response.units),
        positive_count=positive_count,
        eligible_unit_count=len(eligible),
        unassessed_unit_count=len(unassessed),
        selected_count=selected_count,
        selected_eligible_count=selected_eligible_count,
        found=found,
        recall=None if positive_count == 0 else found / positive_count,
        precision=None if selected_eligible_count == 0 else found / selected_eligible_count,
        random_expected_recall=None if positive_count == 0 else selected_count / len(response.units),
        assessment_coverage=len(eligible) / len(response.units),
        selected_assessment_coverage=(
            None if selected_count == 0 else selected_eligible_count / selected_count
        ),
        clean_response=clean,
        fully_assessed=fully_assessed,
    )
