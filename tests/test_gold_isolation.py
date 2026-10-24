from __future__ import annotations

from dataclasses import replace

import pytest

from claimreplay.io import load_response
from claimreplay.metrics import compute_metrics
from claimreplay.models import FrozenAttributes, SupportLevel
from claimreplay.policies import FeatureScorePolicy
from claimreplay.tasks import EvaluationTask, UnassessedHandling, unsupported_detection
from claimreplay.validation import assert_policy_view_gold_free


TOY = "examples/fixtures/toy_response.json"


def test_policy_visible_object_has_no_native_provenance_or_evidence():
    view = load_response(TOY).policy_view()
    assert_policy_view_gold_free(view)
    assert all(not hasattr(unit, "provenance") for unit in view.units)
    assert all(not hasattr(unit, "evidence") for unit in view.units)
    assert view.response_id.startswith("policy-view:")


def test_leakage_scanner_rejects_label_named_observable_feature():
    response = load_response(TOY)
    units = list(response.units)
    units[0] = replace(
        units[0],
        observable_features=FrozenAttributes.from_mapping({"gold_label": "NO_SUPPORT"}),
    )
    leaked = replace(response, units=tuple(units))
    with pytest.raises(ValueError, match="may reveal"):
        assert_policy_view_gold_free(leaked.policy_view())


def test_unassessed_units_are_explicitly_excluded_or_rejected():
    response = load_response(TOY)
    assessments = list(response.gold_assessments)
    assessments[0] = replace(assessments[0], support=SupportLevel.UNASSESSED)
    partial = replace(response, gold_assessments=tuple(assessments))
    metrics = compute_metrics(partial, ("u01", "u03"), unsupported_detection())
    assert metrics.unassessed_unit_count == 1
    assert metrics.eligible_unit_count == 19
    assert metrics.selected_count == 2
    assert metrics.selected_eligible_count == 1
    assert metrics.precision == 1.0
    assert metrics.assessment_coverage == 19 / 20
    assert metrics.selected_assessment_coverage == 0.5
    assert metrics.fully_assessed is False
    strict = EvaluationTask(
        name="strict",
        metric_prefix="Strict",
        positive_labels=frozenset({SupportLevel.NO_SUPPORT}),
        unassessed_handling=UnassessedHandling.ERROR,
    )
    with pytest.raises(ValueError, match="forbids"):
        compute_metrics(partial, ("u03",), strict)
