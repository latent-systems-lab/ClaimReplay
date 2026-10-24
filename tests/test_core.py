from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from examples.custom_policy import LongestUnitFirstPolicy
from claimreplay.io import load_response
from claimreplay.models import SupportLevel
from claimreplay.policies import FeatureScorePolicy, PrivilegedOraclePolicy, RandomPolicy, VerificationPolicy
from claimreplay.replay import replay, resolve_budget
from claimreplay.tasks import review_need_detection, unsupported_detection


TOY = "examples/fixtures/toy_response.json"


def test_policy_view_has_no_gold_assessments():
    response = load_response(TOY)
    view = response.policy_view()
    assert not hasattr(view, "gold_assessments")
    assert not hasattr(view, "gold_hash")
    assert all(not hasattr(unit, "support") for unit in view.units)


def test_frozen_response_cannot_be_mutated():
    response = load_response(TOY)
    with pytest.raises(FrozenInstanceError):
        response.response_id = "changed"
    with pytest.raises(FrozenInstanceError):
        response.units[0].text = "changed"


def test_hand_computed_twenty_unit_case():
    response = load_response(TOY)
    trace = replay(response, FeatureScorePolicy("priority"), unsupported_detection(), 2)
    assert trace.selected_unit_ids == ("u03", "u10")
    assert trace.metrics.positive_count == 2
    assert trace.metrics.found == 2
    assert trace.metrics.recall == 1.0
    assert trace.metrics.precision == 1.0
    assert trace.metrics.random_expected_recall == 0.1
    assert trace.metrics.as_dict()["UnsupportedRecall"] == 1.0
    assert "ReviewNeededRecall" not in trace.metrics.as_dict()


def test_partial_support_is_not_unsupported_but_is_review_needed():
    response = load_response(TOY)
    assessments = list(response.gold_assessments)
    assessments[0] = assessments[0].__class__(unit_id="u01", support=SupportLevel.PARTIAL_SUPPORT)
    response = response.__class__(response.response_id, response.units, tuple(assessments), response.metadata)
    unsupported = replay(response, FeatureScorePolicy("priority"), unsupported_detection(), 2)
    review_needed = replay(response, FeatureScorePolicy("priority"), review_need_detection(), 2)
    assert unsupported.metrics.positive_count == 2
    assert review_needed.metrics.positive_count == 3
    assert "UnsupportedRecall" in unsupported.metrics.as_dict()
    assert "ReviewNeededRecall" in review_needed.metrics.as_dict()


def test_replay_preserves_response_and_gold_hashes():
    response = load_response(TOY)
    replay_1 = replay(response, RandomPolicy(), unsupported_detection(), 2, seed=7)
    replay_2 = replay(response, RandomPolicy(), unsupported_detection(), 2, seed=8)
    assert replay_1.response_hash == replay_2.response_hash == response.response_hash
    assert replay_1.gold_hash == replay_2.gold_hash == response.gold_hash


def test_replay_trace_is_exactly_reproducible_for_fixed_seed():
    response = load_response(TOY)
    first = replay(response, RandomPolicy(), unsupported_detection(), 2, seed=47)
    second = replay(response, RandomPolicy(), unsupported_detection(), 2, seed=47)
    assert first.to_dict() == second.to_dict()
    assert first.triage_cost.model_calls == 0
    assert first.resolved_budget == first.to_dict()["inspection_budget"]


def test_gold_change_does_not_change_response_hash():
    response = load_response(TOY)
    assessments = list(response.gold_assessments)
    assessments[0] = assessments[0].__class__(unit_id="u01", support=SupportLevel.PARTIAL_SUPPORT)
    changed = response.__class__(response.response_id, response.units, tuple(assessments), response.metadata)
    assert changed.response_hash == response.response_hash
    assert changed.gold_hash != response.gold_hash


def test_oracle_is_not_an_ordinary_policy():
    assert not isinstance(PrivilegedOraclePolicy(), VerificationPolicy)
    response = load_response(TOY)
    trace = replay(response, PrivilegedOraclePolicy(), unsupported_detection(), 2)
    assert trace.metrics.recall == 1.0


def test_small_custom_policy_plugin_runs_without_gold_access():
    response = load_response(TOY)
    trace = replay(response, LongestUnitFirstPolicy(), unsupported_detection(), 2)
    assert trace.policy_name == "example:longest-unit-first"
    assert trace.response_hash == response.response_hash


def test_fractional_budgets_use_documented_ceiling():
    assert resolve_budget(0.05, 20) == 1
    assert resolve_budget(0.05, 40) == 2
    assert resolve_budget(0.20, 21) == 5


def test_clean_response_keeps_explicit_na_recall():
    response = load_response(TOY)
    assessments = tuple(
        row.__class__(unit_id=row.unit_id, support=SupportLevel.FULL_SUPPORT)
        for row in response.gold_assessments
    )
    clean = response.__class__(response.response_id, response.units, assessments, response.metadata)
    trace = replay(clean, RandomPolicy(), unsupported_detection(), 2)
    assert trace.metrics.clean_response is True
    assert trace.metrics.recall is None
    assert trace.metrics.random_expected_recall is None
    assert trace.metrics.found == 0
