"""Frozen task adapters for LLM priority scoring and support judging."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .llm import StructuredRequest, content_hash
from .models import PolicyView
from .policies import RankedUnit, TriageCost, VerificationPolicy


PRIORITY_SCHEMA = {
    "type": "object",
    "properties": {"priority_score": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["priority_score"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "support_label": {
            "type": "string",
            "enum": ["FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT", "ABSTAIN"],
        }
    },
    "required": ["support_label"],
    "additionalProperties": False,
}


def validate_priority_output(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) != {"priority_score"}:
        raise ValueError("Priority output must contain only priority_score")
    score = payload["priority_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError("priority_score must be an integer in [0, 100]")
    return {"priority_score": score}


class JudgeLabel(str, Enum):
    FULL_SUPPORT = "FULL_SUPPORT"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    NO_SUPPORT = "NO_SUPPORT"
    ABSTAIN = "ABSTAIN"


def validate_judge_output(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) != {"support_label"}:
        raise ValueError("Judge output must contain only support_label")
    try:
        label = JudgeLabel(str(payload["support_label"]))
    except ValueError as exc:
        raise ValueError("Unknown judge support label") from exc
    return {"support_label": label.value}


def priority_request(unit_id: str, text: str, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> StructuredRequest:
    return StructuredRequest(
        request_id=unit_id,
        task="claim_review_priority",
        system_prompt=prompt,
        user_payload={"claim_text": text},
        schema_name="claim_review_priority",
        json_schema=PRIORITY_SCHEMA,
        metadata=metadata or {},
    )


def judge_request(
    unit_id: str,
    claim_text: str,
    evidence_text: str,
    prompt: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> StructuredRequest:
    if not evidence_text.strip():
        raise ValueError("Judge requests require nonempty evidence text")
    return StructuredRequest(
        request_id=unit_id,
        task="citation_support_judgment",
        system_prompt=prompt,
        user_payload={"claim_text": claim_text, "evidence_text": evidence_text},
        schema_name="citation_support_judgment",
        json_schema=JUDGE_SCHEMA,
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class RecordedPriorityScore:
    unit_id: str
    text_sha256: str
    score: int


class RecordedLlmPriorityPolicy(VerificationPolicy):
    """Replay previously acquired pointwise scores without API access."""

    name = "llm:gpt-5.4-mini-review-priority-v1"

    def __init__(self, scores: Mapping[str, RecordedPriorityScore]) -> None:
        self.scores = dict(scores)

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        rows = []
        for unit in response.units:
            record = self.scores.get(unit.unit_id)
            if record is None:
                raise ValueError(f"Missing LLM priority score for {unit.unit_id}")
            if record.text_sha256 != content_hash(unit.text):
                raise ValueError(f"LLM priority score input mismatch for {unit.unit_id}")
            if not 0 <= record.score <= 100:
                raise ValueError(f"Invalid LLM priority score for {unit.unit_id}")
            rows.append(RankedUnit(unit.unit_id, float(record.score)))
        return tuple(sorted(rows, key=lambda row: (-float(row.score), row.unit_id)))

    def triage_cost(self, response: PolicyView) -> TriageCost:
        return TriageCost(
            units_considered=len(response.units),
            score_evaluations=len(response.units),
            model_calls=0,
            cache_hits=len(response.units),
        )
