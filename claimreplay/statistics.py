"""Paired cluster-aware uncertainty estimates for replay traces.

The response is the pairing unit. Cluster bootstrap dimensions are selected
explicitly by the analyst. For crossed TREC topic/run dependence, pass
``cluster_keys=("topic_id", "run_id")``; independent resampling of each
dimension produces pigeonhole-bootstrap multiplicity weights.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Iterable

from .replay import ReplayTrace


@dataclass(frozen=True)
class PairedDifference:
    response_hash: str
    task_name: str
    resolved_budget: int
    seed: int
    cluster_ids: tuple[tuple[str, str], ...]
    baseline_value: float
    contender_value: float

    @property
    def difference(self) -> float:
        return self.contender_value - self.baseline_value


@dataclass(frozen=True)
class ClusterBootstrapResult:
    metric: str
    baseline_policy: str
    contender_policy: str
    estimate: float
    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float
    paired_response_count: int
    cluster_keys: tuple[str, ...]
    cluster_cardinalities: tuple[int, ...]
    resamples: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "contrast": "contender_minus_baseline",
            "baseline_policy": self.baseline_policy,
            "contender_policy": self.contender_policy,
            "estimate": self.estimate,
            "confidence_level": self.confidence_level,
            "confidence_interval": [self.confidence_interval_low, self.confidence_interval_high],
            "paired_response_count": self.paired_response_count,
            "cluster_keys": list(self.cluster_keys),
            "cluster_cardinalities": list(self.cluster_cardinalities),
            "resamples": self.resamples,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TopicMacroBootstrapResult:
    """Paired difference in topic-macro positive-unit recall.

    Within each topic, found positives and all positives are pooled across the
    observed runs. Topics, rather than topic-run responses, then receive equal
    weight. Runs are treated as a fixed evaluation panel; uncertainty is over
    the observed topic population.
    """

    metric: str
    baseline_policy: str
    contender_policy: str
    estimate: float
    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float
    paired_response_count: int
    topic_count: int
    estimable_topic_count: int
    zero_positive_topic_count: int
    resamples: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "estimand": "topic_macro_pooled_positive_unit_recall_difference",
            "contrast": "contender_minus_baseline",
            "baseline_policy": self.baseline_policy,
            "contender_policy": self.contender_policy,
            "estimate": self.estimate,
            "confidence_level": self.confidence_level,
            "confidence_interval": [self.confidence_interval_low, self.confidence_interval_high],
            "paired_response_count": self.paired_response_count,
            "topic_count": self.topic_count,
            "estimable_topic_count": self.estimable_topic_count,
            "zero_positive_topic_count": self.zero_positive_topic_count,
            "resamples": self.resamples,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SmallSampleMeanInterval:
    estimate: float
    standard_error: float
    degrees_of_freedom: int
    critical_value: float
    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float
    cluster_count: int

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "degrees_of_freedom": self.degrees_of_freedom,
            "critical_value": self.critical_value,
            "confidence_level": self.confidence_level,
            "confidence_interval": [self.confidence_interval_low, self.confidence_interval_high],
            "cluster_count": self.cluster_count,
            "method": "student_t_interval_over_cluster_effects",
        }


def leave_one_cluster_out_means(cluster_effects: dict[str, float]) -> dict[str, float]:
    """Return the mean effect after omitting each named cluster once."""
    if len(cluster_effects) < 2:
        raise ValueError("At least two cluster effects are required")
    values = {str(key): float(value) for key, value in cluster_effects.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Cluster effects must be finite")
    return {
        omitted: fmean(value for key, value in values.items() if key != omitted)
        for omitted in sorted(values)
    }


_T_975 = {
    1: 12.706204736, 2: 4.30265273, 3: 3.182446305, 4: 2.776445105,
    5: 2.570581836, 6: 2.446911851, 7: 2.364624252, 8: 2.306004135,
    9: 2.262157163, 10: 2.228138852, 11: 2.20098516, 12: 2.17881283,
    13: 2.160368656, 14: 2.144786688, 15: 2.131449546, 16: 2.119905299,
    17: 2.109815578, 18: 2.10092204, 19: 2.093024054, 20: 2.085963447,
    21: 2.079613845, 22: 2.073873068, 23: 2.06865761, 24: 2.063898562,
    25: 2.059538553, 26: 2.055529439, 27: 2.051830516, 28: 2.048407142,
    29: 2.045229642, 30: 2.042272456,
}


def analytic_random_expected_found(positive_count: int | float, budget: int, unit_count: int) -> float:
    """Expected positives in a uniform size-``budget`` sample without replacement."""
    if unit_count <= 0 or budget < 0 or budget > unit_count:
        raise ValueError("Require unit_count > 0 and 0 <= budget <= unit_count")
    if positive_count < 0 or positive_count > unit_count:
        raise ValueError("positive_count must be between zero and unit_count")
    return float(positive_count) * budget / unit_count


def small_sample_t_interval(
    cluster_effects: Iterable[float], *, confidence_level: float = 0.95
) -> SmallSampleMeanInterval:
    """Student-t interval over independent cluster-level effect estimates.

    The built-in critical-value table intentionally supports only the frozen
    95% protocol and at most 31 clusters; this prevents silently substituting a
    normal approximation in the nine-topic primary analysis.
    """
    values = [float(value) for value in cluster_effects]
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("At least two finite cluster effects are required")
    if confidence_level != 0.95:
        raise ValueError("Only the prespecified 95% interval is supported")
    df = len(values) - 1
    if df not in _T_975:
        raise ValueError("Built-in exact critical values support 2 through 31 clusters")
    estimate = fmean(values)
    sample_variance = sum((value - estimate) ** 2 for value in values) / df
    standard_error = math.sqrt(sample_variance / len(values))
    margin = _T_975[df] * standard_error
    return SmallSampleMeanInterval(
        estimate=estimate,
        standard_error=standard_error,
        degrees_of_freedom=df,
        critical_value=_T_975[df],
        confidence_level=confidence_level,
        confidence_interval_low=estimate - margin,
        confidence_interval_high=estimate + margin,
        cluster_count=len(values),
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down family-wise-error adjustment."""
    for name, value in p_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid p-value for {name!r}")
    ordered = sorted(p_values.items(), key=lambda row: row[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def paired_sign_flip_pvalue(
    differences: Iterable[float], *, seed: int = 0, resamples: int = 100000
) -> float:
    """Two-sided paired randomization p-value over independent cluster effects.

    Enumeration is exact for at most 20 non-zero clusters. Larger inputs use a
    deterministic Monte Carlo estimate with the standard plus-one correction.
    """
    values = [float(value) for value in differences if value != 0.0]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Differences must be finite")
    if not values:
        return 1.0
    observed = abs(fmean(values))
    tolerance = 1e-15
    if len(values) <= 20:
        exceed = 0
        total = 1 << len(values)
        for mask in range(total):
            statistic = abs(sum(value if mask & (1 << index) else -value for index, value in enumerate(values)) / len(values))
            exceed += statistic + tolerance >= observed
        return exceed / total
    if resamples < 1:
        raise ValueError("resamples must be positive")
    generator = random.Random(seed)
    exceed = 0
    for _ in range(resamples):
        statistic = abs(fmean(value if generator.random() < 0.5 else -value for value in values))
        exceed += statistic + tolerance >= observed
    return (exceed + 1) / (resamples + 1)


def _pair_key(trace: ReplayTrace) -> tuple[str, str, int, int]:
    return (trace.response_hash, trace.task_name, trace.resolved_budget, trace.seed)


def _numeric_metric(trace: ReplayTrace, metric: str) -> float | None:
    value = trace.metrics.as_dict().get(metric)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Metric {metric!r} is not a finite numeric replay metric")
    return float(value)


def pair_policy_traces(
    traces: Iterable[ReplayTrace],
    *,
    baseline_policy: str,
    contender_policy: str,
    metric: str,
    cluster_keys: tuple[str, ...],
) -> tuple[PairedDifference, ...]:
    if not cluster_keys or len(cluster_keys) != len(set(cluster_keys)):
        raise ValueError("cluster_keys must contain one or more unique keys")
    relevant = [
        trace for trace in traces if trace.policy_name in {baseline_policy, contender_policy}
    ]
    grouped: dict[tuple[str, str, int, int], dict[str, ReplayTrace]] = {}
    for trace in relevant:
        policies = grouped.setdefault(_pair_key(trace), {})
        if trace.policy_name in policies:
            raise ValueError(f"Duplicate trace for policy {trace.policy_name!r} and pairing key")
        policies[trace.policy_name] = trace
    if not grouped:
        raise ValueError("No traces match the requested policies")

    output = []
    for key, policies in sorted(grouped.items()):
        if set(policies) != {baseline_policy, contender_policy}:
            raise ValueError(f"Unpaired policy trace for response hash {key[0]}")
        baseline = policies[baseline_policy]
        contender = policies[contender_policy]
        baseline_clusters = baseline.cluster_ids.to_dict()
        contender_clusters = contender.cluster_ids.to_dict()
        clusters = []
        for cluster_key in cluster_keys:
            left = baseline_clusters.get(cluster_key)
            right = contender_clusters.get(cluster_key)
            if left is None or right is None:
                raise ValueError(f"Missing required cluster ID {cluster_key!r} for response hash {key[0]}")
            if left != right:
                raise ValueError(f"Cluster ID {cluster_key!r} differs within a paired response")
            clusters.append((cluster_key, str(left)))
        baseline_value = _numeric_metric(baseline, metric)
        contender_value = _numeric_metric(contender, metric)
        if baseline_value is None or contender_value is None:
            continue
        output.append(PairedDifference(
            response_hash=key[0],
            task_name=key[1],
            resolved_budget=key[2],
            seed=key[3],
            cluster_ids=tuple(clusters),
            baseline_value=baseline_value,
            contender_value=contender_value,
        ))
    if not output:
        raise ValueError(f"No complete pairs have a defined value for metric {metric!r}")
    return tuple(output)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(
    traces: Iterable[ReplayTrace],
    *,
    baseline_policy: str,
    contender_policy: str,
    metric: str,
    cluster_keys: tuple[str, ...],
    resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ClusterBootstrapResult:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    pairs = pair_policy_traces(
        traces,
        baseline_policy=baseline_policy,
        contender_policy=contender_policy,
        metric=metric,
        cluster_keys=cluster_keys,
    )
    levels = [sorted({dict(pair.cluster_ids)[key] for pair in pairs}) for key in cluster_keys]
    if any(len(values) < 2 for values in levels):
        raise ValueError("Each bootstrap cluster dimension requires at least two observed levels")

    generator = random.Random(seed)
    draws: list[float] = []
    attempts = 0
    max_attempts = resamples * 20
    while len(draws) < resamples and attempts < max_attempts:
        attempts += 1
        multiplicities = [
            Counter(generator.choice(values) for _ in range(len(values))) for values in levels
        ]
        weighted_sum = 0.0
        total_weight = 0
        for pair in pairs:
            cluster_map = dict(pair.cluster_ids)
            weight = math.prod(
                multiplicities[index][cluster_map[key]] for index, key in enumerate(cluster_keys)
            )
            weighted_sum += weight * pair.difference
            total_weight += weight
        if total_weight:
            draws.append(weighted_sum / total_weight)
    if len(draws) != resamples:
        raise RuntimeError("Could not obtain enough non-empty clustered bootstrap samples")

    alpha = (1.0 - confidence_level) / 2.0
    return ClusterBootstrapResult(
        metric=metric,
        baseline_policy=baseline_policy,
        contender_policy=contender_policy,
        estimate=fmean(pair.difference for pair in pairs),
        confidence_level=confidence_level,
        confidence_interval_low=_quantile(draws, alpha),
        confidence_interval_high=_quantile(draws, 1.0 - alpha),
        paired_response_count=len(pairs),
        cluster_keys=cluster_keys,
        cluster_cardinalities=tuple(len(values) for values in levels),
        resamples=resamples,
        seed=seed,
    )


def paired_topic_macro_bootstrap(
    traces: Iterable[ReplayTrace],
    *,
    baseline_policy: str,
    contender_policy: str,
    found_metric: str,
    positive_count_metric: str,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> TopicMacroBootstrapResult:
    """Bootstrap equally weighted topic-level pooled-recall contrasts.

    Every paired response remains in the inspection population. A topic with
    no positive assessment under the selected task has undefined recall and is
    reported, not silently converted to zero. The positive denominator must be
    identical across policies because assessments are evaluator-only.
    """
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    relevant = [trace for trace in traces if trace.policy_name in {baseline_policy, contender_policy}]
    grouped: dict[tuple[str, str, int, int], dict[str, ReplayTrace]] = {}
    for trace in relevant:
        policies = grouped.setdefault(_pair_key(trace), {})
        if trace.policy_name in policies:
            raise ValueError(f"Duplicate trace for policy {trace.policy_name!r} and pairing key")
        policies[trace.policy_name] = trace
    if not grouped:
        raise ValueError("No traces match the requested policies")

    topic_totals: dict[str, list[float]] = {}
    pair_count = 0
    for key, policies in sorted(grouped.items()):
        if set(policies) != {baseline_policy, contender_policy}:
            raise ValueError(f"Unpaired policy trace for response hash {key[0]}")
        baseline = policies[baseline_policy]
        contender = policies[contender_policy]
        baseline_topic = baseline.cluster_ids.to_dict().get("topic_id")
        contender_topic = contender.cluster_ids.to_dict().get("topic_id")
        if baseline_topic is None or contender_topic is None or str(baseline_topic) != str(contender_topic):
            raise ValueError(f"Missing or inconsistent topic_id for response hash {key[0]}")
        baseline_found = _numeric_metric(baseline, found_metric)
        contender_found = _numeric_metric(contender, found_metric)
        baseline_positive = _numeric_metric(baseline, positive_count_metric)
        contender_positive = _numeric_metric(contender, positive_count_metric)
        if None in (baseline_found, contender_found, baseline_positive, contender_positive):
            raise ValueError("Found and positive-count metrics must be defined for every paired response")
        if baseline_positive != contender_positive:
            raise ValueError("Positive-count denominator differs across paired policies")
        totals = topic_totals.setdefault(str(baseline_topic), [0.0, 0.0, 0.0])
        totals[0] += float(baseline_found)
        totals[1] += float(contender_found)
        totals[2] += float(baseline_positive)
        pair_count += 1

    differences = {
        topic: (values[1] / values[2]) - (values[0] / values[2])
        for topic, values in topic_totals.items()
        if values[2] > 0
    }
    if not differences:
        raise ValueError("No topic has a positive unit for the selected evaluation task")
    topics = sorted(differences)
    generator = random.Random(seed)
    draws = [
        fmean(differences[generator.choice(topics)] for _ in range(len(topics)))
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return TopicMacroBootstrapResult(
        metric=f"topic_macro_{found_metric}_over_{positive_count_metric}",
        baseline_policy=baseline_policy,
        contender_policy=contender_policy,
        estimate=fmean(differences.values()),
        confidence_level=confidence_level,
        confidence_interval_low=_quantile(draws, alpha),
        confidence_interval_high=_quantile(draws, 1.0 - alpha),
        paired_response_count=pair_count,
        topic_count=len(topic_totals),
        estimable_topic_count=len(differences),
        zero_positive_topic_count=len(topic_totals) - len(differences),
        resamples=resamples,
        seed=seed,
    )


def topic_macro_policy_vs_random_expectation_bootstrap(
    traces: Iterable[ReplayTrace],
    *,
    contender_policy: str,
    found_metric: str,
    positive_count_metric: str,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> TopicMacroBootstrapResult:
    """Compare a policy with the exact uniform-random recall expectation.

    For a response with ``b`` selected of ``n`` units and ``p`` positives, the
    expected number found under uniform sampling without replacement is
    ``p*b/n``. This removes arbitrary random-policy seed noise from the primary
    recall contrast while retaining executable random traces for diagnostics.
    """
    rows = [trace for trace in traces if trace.policy_name == contender_policy]
    if not rows:
        raise ValueError("No traces match the requested contender policy")
    seen = set()
    topic_totals: dict[str, list[float]] = {}
    for trace in rows:
        key = _pair_key(trace)
        if key in seen:
            raise ValueError("Duplicate contender trace for pairing key")
        seen.add(key)
        topic = trace.cluster_ids.to_dict().get("topic_id")
        if topic is None:
            raise ValueError("Missing topic_id")
        found = _numeric_metric(trace, found_metric)
        positive = _numeric_metric(trace, positive_count_metric)
        total = trace.metrics.total_unit_count
        if found is None or positive is None or total <= 0:
            raise ValueError("Required metrics must be defined and response non-empty")
        values = topic_totals.setdefault(str(topic), [0.0, 0.0, 0.0])
        values[0] += float(found)
        values[1] += analytic_random_expected_found(float(positive), trace.resolved_budget, total)
        values[2] += float(positive)
    differences = {
        topic: (values[0] - values[1]) / values[2]
        for topic, values in topic_totals.items() if values[2] > 0
    }
    if not differences:
        raise ValueError("No topic has a positive unit for the selected evaluation task")
    topics = sorted(differences)
    generator = random.Random(seed)
    draws = [fmean(differences[generator.choice(topics)] for _ in topics) for _ in range(resamples)]
    alpha = (1.0 - confidence_level) / 2.0
    return TopicMacroBootstrapResult(
        metric=f"topic_macro_{found_metric}_over_{positive_count_metric}",
        baseline_policy="random:analytic-expectation",
        contender_policy=contender_policy,
        estimate=fmean(differences.values()),
        confidence_level=confidence_level,
        confidence_interval_low=_quantile(draws, alpha),
        confidence_interval_high=_quantile(draws, 1.0 - alpha),
        paired_response_count=len(rows),
        topic_count=len(topic_totals),
        estimable_topic_count=len(differences),
        zero_positive_topic_count=len(topic_totals) - len(differences),
        resamples=resamples,
        seed=seed,
    )


def topic_effects_policy_vs_random_expectation(
    traces: Iterable[ReplayTrace],
    *,
    contender_policy: str,
    found_metric: str,
    positive_count_metric: str,
) -> dict[str, float | None]:
    """Return every topic effect, retaining zero-positive topics as ``None``."""
    rows = [trace for trace in traces if trace.policy_name == contender_policy]
    if not rows:
        raise ValueError("No traces match the requested contender policy")
    seen = set()
    totals: dict[str, list[float]] = {}
    for trace in rows:
        key = _pair_key(trace)
        if key in seen:
            raise ValueError("Duplicate contender trace for pairing key")
        seen.add(key)
        topic = trace.cluster_ids.to_dict().get("topic_id")
        if topic is None:
            raise ValueError("Missing topic_id")
        found = _numeric_metric(trace, found_metric)
        positive = _numeric_metric(trace, positive_count_metric)
        if found is None or positive is None:
            raise ValueError("Required metrics must be defined")
        values = totals.setdefault(str(topic), [0.0, 0.0, 0.0])
        values[0] += float(found)
        values[1] += analytic_random_expected_found(
            float(positive), trace.resolved_budget, trace.metrics.total_unit_count
        )
        values[2] += float(positive)
    return {
        topic: None if values[2] == 0 else (values[0] - values[1]) / values[2]
        for topic, values in sorted(totals.items())
    }
