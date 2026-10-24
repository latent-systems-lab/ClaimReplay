"""Fixed-response replay engine and inspectable replay trace."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .metrics import ReplayMetrics, compute_metrics
from .models import FrozenAttributes, FrozenResponse
from .policies import (
    EvidenceAwareVerificationPolicy,
    PrivilegedOraclePolicy,
    RankedUnit,
    TriageCost,
    VerificationPolicy,
)
from .tasks import EvaluationTask
from .validation import assert_policy_view_gold_free


@dataclass(frozen=True)
class ReplayTrace:
    response_id: str
    response_hash: str
    gold_hash: str
    policy_name: str
    task_name: str
    requested_budget: int | float
    resolved_budget: int
    seed: int
    ranking: tuple[RankedUnit, ...]
    selected_unit_ids: tuple[str, ...]
    triage_cost: TriageCost
    cluster_ids: FrozenAttributes
    metrics: ReplayMetrics

    def to_dict(self) -> dict:
        return {
            "response_id": self.response_id,
            "response_hash": self.response_hash,
            "gold_hash": self.gold_hash,
            "policy": self.policy_name,
            "task": self.task_name,
            "requested_budget": self.requested_budget,
            "resolved_budget": self.resolved_budget,
            "seed": self.seed,
            "ranking": [row.__dict__ for row in self.ranking],
            "selected_unit_ids": list(self.selected_unit_ids),
            "inspection_budget": self.resolved_budget,
            "triage_cost": self.triage_cost.to_dict(),
            "cluster_ids": self.cluster_ids.to_dict(),
            "metrics": self.metrics.as_dict(),
        }


def resolve_budget(budget: int | float, unit_count: int) -> int:
    if isinstance(budget, bool):
        raise ValueError("Budget must be an integer count or fraction")
    if isinstance(budget, int):
        resolved = budget
    elif isinstance(budget, float) and 0.0 < budget <= 1.0:
        resolved = math.ceil(budget * unit_count)
    else:
        raise ValueError("Fractional budget must be in (0, 1]")
    if not 0 <= resolved <= unit_count:
        raise ValueError("Resolved budget must be between zero and response size")
    return resolved


def replay(
    response: FrozenResponse,
    policy: VerificationPolicy | EvidenceAwareVerificationPolicy | PrivilegedOraclePolicy,
    task: EvaluationTask,
    budget: int | float,
    *,
    seed: int = 0,
) -> ReplayTrace:
    before_response_hash = response.response_hash
    before_gold_hash = response.gold_hash
    if isinstance(policy, PrivilegedOraclePolicy):
        ranking = policy.rank(response, task, seed=seed)
        triage_cost = policy.triage_cost(response)
    elif isinstance(policy, EvidenceAwareVerificationPolicy):
        policy_view = response.evidence_policy_view()
        assert_policy_view_gold_free(policy_view)
        ranking = policy.rank(policy_view, seed=seed)
        triage_cost = policy.triage_cost(policy_view)
    elif isinstance(policy, VerificationPolicy):
        policy_view = response.policy_view()
        assert_policy_view_gold_free(
            policy_view,
            observable_feature_schema=policy.observable_feature_schema(),
        )
        ranking = policy.rank(policy_view, seed=seed)
        triage_cost = policy.triage_cost(policy_view)
    else:
        raise TypeError("Policy must implement an ordinary, evidence-aware, or privileged policy interface")
    ranked_ids = [row.unit_id for row in ranking]
    expected_ids = {unit.unit_id for unit in response.units}
    if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != expected_ids:
        raise ValueError("A policy must rank every unit exactly once")
    resolved = resolve_budget(budget, len(response.units))
    selected = tuple(ranked_ids[:resolved])
    metrics = compute_metrics(response, selected, task)
    cluster_ids = {
        key: response.metadata.get(key)
        for key in ("dataset", "topic_id", "run_id", "source_pool_id", "group_id", "manifest_id")
        if response.metadata.get(key) is not None
    }
    if response.response_hash != before_response_hash or response.gold_hash != before_gold_hash:
        raise RuntimeError("Frozen response or gold assessments changed during replay")
    return ReplayTrace(
        response_id=response.response_id,
        response_hash=before_response_hash,
        gold_hash=before_gold_hash,
        policy_name=policy.name,
        task_name=task.name,
        requested_budget=budget,
        resolved_budget=resolved,
        seed=seed,
        ranking=ranking,
        selected_unit_ids=selected,
        triage_cost=triage_cost,
        cluster_ids=FrozenAttributes.from_mapping(cluster_ids),
        metrics=metrics,
    )
