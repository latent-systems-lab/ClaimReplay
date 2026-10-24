from __future__ import annotations

import pytest

from claimreplay.judge_metrics import (
    agreement_metrics,
    binary_agreement_metrics,
    matched_fixed_selection_recall,
    topic_bootstrap_interval,
)


def test_multiclass_agreement_exact_case():
    labels = ["FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT"]
    result = agreement_metrics(labels, labels)
    assert result["exact_agreement"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["cohen_kappa"] == 1.0


def test_binary_mapping_keeps_partial_separate_in_multiclass_but_positive_in_task():
    result = binary_agreement_metrics(
        ["FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT"],
        ["FULL_SUPPORT", "NO_SUPPORT", "PARTIAL_SUPPORT"],
    )
    assert result["exact_agreement"] == 1.0


def test_fixed_selection_keeps_unmatched_slots_and_does_not_reallocate():
    units = [
        {"unit_id": "a", "human_label": "NO_SUPPORT", "llm_label": "NO_SUPPORT"},
        {"unit_id": "b", "human_label": "FULL_SUPPORT", "llm_label": "ABSTAIN"},
        {"unit_id": "c", "human_label": "PARTIAL_SUPPORT", "llm_label": "FULL_SUPPORT"},
    ]
    result = matched_fixed_selection_recall(units, ["a", "b"], label_field="human_label")
    assert result == {
        "positive_count": 2,
        "found": 1,
        "recall": .5,
        "selected_total": 2,
        "selected_matched": 1,
        "matched_selection_coverage": .5,
    }


def test_topic_bootstrap_is_deterministic_and_requires_topics():
    first = topic_bootstrap_interval({"t1": .2, "t2": .4, "t3": .8}, resamples=1000)
    second = topic_bootstrap_interval({"t1": .2, "t2": .4, "t3": .8}, resamples=1000)
    assert first == second
    assert first["estimate"] == pytest.approx((.2 + .4 + .8) / 3)
    with pytest.raises(ValueError):
        topic_bootstrap_interval({"t1": .2})


def test_macro_f1_includes_zero_true_positive_classes():
    result = agreement_metrics(
        ["FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT"],
        ["FULL_SUPPORT", "FULL_SUPPORT", "FULL_SUPPORT"],
    )
    assert result["per_class"]["PARTIAL_SUPPORT"]["f1"] == 0
    assert result["per_class"]["NO_SUPPORT"]["f1"] == 0
    assert result["macro_f1"] == pytest.approx((0.5 + 0 + 0) / 3)


def test_macro_f1_all_wrong_is_zero():
    result = agreement_metrics(
        ["FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT"],
        ["PARTIAL_SUPPORT", "NO_SUPPORT", "FULL_SUPPORT"],
    )
    assert result["macro_f1"] == 0
