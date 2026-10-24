from __future__ import annotations

from dataclasses import replace

import pytest

from claimreplay.io import load_response
from claimreplay.models import FrozenAttributes, SupportLevel
from claimreplay.policies import FeatureScorePolicy
from claimreplay.replay import replay
from claimreplay.statistics import (
    holm_adjust,
    paired_cluster_bootstrap,
    paired_topic_macro_bootstrap,
    pair_policy_traces,
    paired_sign_flip_pvalue,
    topic_macro_policy_vs_random_expectation_bootstrap,
)
from claimreplay.tasks import unsupported_detection


def crossed_traces():
    base = load_response("examples/fixtures/toy_response.json")
    output = []
    for topic in ("t1", "t2"):
        for run in ("r1", "r2"):
            units = []
            for unit in base.units:
                is_positive = unit.unit_id in {"u03", "u10"}
                units.append(replace(
                    unit,
                    observable_features=FrozenAttributes.from_mapping({
                        "good_score": 1 if is_positive else 0,
                        "bad_score": 0 if is_positive else 1,
                    }),
                ))
            response = replace(
                base,
                response_id=f"{topic}:{run}",
                units=tuple(units),
                metadata=FrozenAttributes.from_mapping({
                    "dataset": "synthetic-crossed",
                    "topic_id": topic,
                    "run_id": run,
                    "policy_visible": {},
                }),
            )
            output.append(replay(
                response,
                FeatureScorePolicy("bad_score", name="baseline"),
                unsupported_detection(),
                2,
            ))
            output.append(replay(
                response,
                FeatureScorePolicy("good_score", name="contender"),
                unsupported_detection(),
                2,
            ))
    return output


def test_two_way_paired_cluster_bootstrap_is_deterministic():
    first = paired_cluster_bootstrap(
        crossed_traces(),
        baseline_policy="baseline",
        contender_policy="contender",
        metric="UnsupportedRecall",
        cluster_keys=("topic_id", "run_id"),
        resamples=200,
        seed=47,
    )
    second = paired_cluster_bootstrap(
        crossed_traces(),
        baseline_policy="baseline",
        contender_policy="contender",
        metric="UnsupportedRecall",
        cluster_keys=("topic_id", "run_id"),
        resamples=200,
        seed=47,
    )
    assert first == second
    assert first.estimate == 1.0
    assert first.confidence_interval_low == 1.0
    assert first.confidence_interval_high == 1.0
    assert first.paired_response_count == 4
    assert first.cluster_cardinalities == (2, 2)


def test_pairing_rejects_missing_policy_counterpart():
    traces = crossed_traces()
    with pytest.raises(ValueError, match="Unpaired"):
        pair_policy_traces(
            traces[:-1],
            baseline_policy="baseline",
            contender_policy="contender",
            metric="UnsupportedRecall",
            cluster_keys=("topic_id",),
        )


def test_topic_macro_bootstrap_equal_weights_topics_and_reports_zero_positive_topic():
    base = load_response("examples/fixtures/toy_response.json")
    traces = []
    for topic, runs, zero_positive in (("many-runs", range(5), False), ("zero-positive", range(1), True)):
        for run in runs:
            if zero_positive:
                gold = tuple(replace(row, support=SupportLevel.FULL_SUPPORT, raw_label="FULL") for row in base.gold_assessments)
            else:
                gold = base.gold_assessments
            units = tuple(replace(
                unit,
                observable_features=FrozenAttributes.from_mapping({
                    "bad": 0 if unit.unit_id in {"u03", "u10"} else 1,
                    "good": 1 if unit.unit_id in {"u03", "u10"} else 0,
                }),
            ) for unit in base.units)
            response = replace(
                base,
                response_id=f"{topic}:{run}",
                units=units,
                gold_assessments=gold,
                metadata=FrozenAttributes.from_mapping({
                    "dataset": "adversarial",
                    "topic_id": topic,
                    "run_id": f"r{run}",
                    "policy_visible": {},
                }),
            )
            traces.extend((
                replay(response, FeatureScorePolicy("bad", name="baseline"), unsupported_detection(), 2),
                replay(response, FeatureScorePolicy("good", name="contender"), unsupported_detection(), 2),
            ))
    result = paired_topic_macro_bootstrap(
        traces,
        baseline_policy="baseline",
        contender_policy="contender",
        found_metric="UnsupportedFound",
        positive_count_metric="UnsupportedCount",
        resamples=50,
        seed=47,
    )
    assert result.estimate == 1.0
    assert result.topic_count == 2
    assert result.estimable_topic_count == 1
    assert result.zero_positive_topic_count == 1
    assert result.paired_response_count == 6


def test_topic_macro_bootstrap_rejects_unpaired_trace():
    with pytest.raises(ValueError, match="Unpaired"):
        paired_topic_macro_bootstrap(
            crossed_traces()[:-1],
            baseline_policy="baseline",
            contender_policy="contender",
            found_metric="UnsupportedFound",
            positive_count_metric="UnsupportedCount",
            resamples=10,
        )


def test_holm_adjust_is_monotone_in_test_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)


def test_analytic_random_expectation_removes_seed_noise():
    result = topic_macro_policy_vs_random_expectation_bootstrap(
        crossed_traces(),
        contender_policy="contender",
        found_metric="UnsupportedFound",
        positive_count_metric="UnsupportedCount",
        resamples=25,
        seed=47,
    )
    assert result.baseline_policy == "random:analytic-expectation"
    assert result.estimate == pytest.approx(0.9)
    assert result.confidence_interval_low == pytest.approx(0.9)


def test_exact_paired_sign_flip_and_zero_effect():
    assert paired_sign_flip_pvalue([1.0, 1.0, 1.0]) == pytest.approx(0.25)
    assert paired_sign_flip_pvalue([0.0, 0.0]) == 1.0
