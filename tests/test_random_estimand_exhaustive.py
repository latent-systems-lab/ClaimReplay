from __future__ import annotations

from itertools import combinations
from statistics import fmean

import pytest

from claimreplay.statistics import (
    analytic_random_expected_found,
    leave_one_cluster_out_means,
    small_sample_t_interval,
)


@pytest.mark.parametrize("n", range(1, 9))
def test_uniform_without_replacement_expectation_exhaustive(n: int):
    universe = tuple(range(n))
    for positive_count in range(n + 1):
        positives = set(universe[:positive_count])
        for budget in range(n + 1):
            selections = tuple(combinations(universe, budget))
            found = [len(positives.intersection(selection)) for selection in selections]
            expected_found = analytic_random_expected_found(positive_count, budget, n)
            assert fmean(found) == pytest.approx(expected_found)
            if positive_count > 0:
                assert fmean(value / positive_count for value in found) == pytest.approx(budget / n)


def test_topic_pooling_requires_positive_weighted_expected_counts():
    # Response A: P=1,n=2,b=1; response B: P=3,n=6,b=2.
    # Pooling expected counts gives (0.5 + 1.0) / 4 = 0.375.
    pooled = (
        analytic_random_expected_found(1, 1, 2)
        + analytic_random_expected_found(3, 2, 6)
    ) / 4
    assert pooled == pytest.approx(0.375)
    # An unweighted mean of response recalls answers a different estimand.
    assert fmean((1 / 2, 2 / 6)) != pytest.approx(pooled)


def test_zero_positive_recall_is_undefined_but_expected_found_is_zero():
    assert analytic_random_expected_found(0, 3, 10) == 0.0


def test_nine_topic_t_interval_uses_eight_degrees_of_freedom():
    result = small_sample_t_interval([-0.2, -0.1, 0.0, 0.1, 0.2, 0.05, -0.05, 0.15, -0.15])
    assert result.cluster_count == 9
    assert result.degrees_of_freedom == 8
    assert result.critical_value == pytest.approx(2.306004135)
    assert result.confidence_interval_low < result.estimate < result.confidence_interval_high


def test_leave_one_topic_out_influence_is_named_and_exact():
    result = leave_one_cluster_out_means({"t1": 0.0, "t2": 0.3, "t3": 0.6})
    assert result == pytest.approx({"t1": 0.45, "t2": 0.3, "t3": 0.15})
