"""Immutable canonical records and the policy/gold separation boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FrozenAttributes:
    """Deeply immutable JSON-compatible mapping stored in canonical form."""

    entries: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, JsonValue] | None = None) -> "FrozenAttributes":
        values = values or {}
        return cls(tuple(sorted((str(key), _canonical_json(value)) for key, value in values.items())))

    def to_dict(self) -> dict[str, JsonValue]:
        return {key: json.loads(value) for key, value in self.entries}

    def get(self, key: str, default: Any = None) -> Any:
        for candidate, value in self.entries:
            if candidate == key:
                return json.loads(value)
        return default


class SupportLevel(str, Enum):
    FULL_SUPPORT = "FULL_SUPPORT"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    NO_SUPPORT = "NO_SUPPORT"
    UNASSESSED = "UNASSESSED"


@dataclass(frozen=True)
class Provenance:
    dataset: str
    source_id: str | None = None
    citation_ids: tuple[str, ...] = ()
    metadata: FrozenAttributes = field(default_factory=FrozenAttributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_id": self.source_id,
            "citation_ids": list(self.citation_ids),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    text: str | None = None
    source_uri: str | None = None
    metadata: FrozenAttributes = field(default_factory=FrozenAttributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source_uri": self.source_uri,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    items: tuple[EvidenceItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}


@dataclass(frozen=True)
class VerificationUnit:
    unit_id: str
    text: str
    provenance: Provenance
    observable_features: FrozenAttributes = field(default_factory=FrozenAttributes)
    evidence: EvidenceBundle | None = None

    def __post_init__(self) -> None:
        if not self.unit_id or not self.text.strip():
            raise ValueError("Verification units require a non-empty ID and text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "provenance": self.provenance.to_dict(),
            "observable_features": self.observable_features.to_dict(),
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass(frozen=True)
class GoldAssessment:
    unit_id: str
    support: SupportLevel
    raw_label: str | None = None
    assessor: str | None = None
    metadata: FrozenAttributes = field(default_factory=FrozenAttributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "support": self.support.value,
            "raw_label": self.raw_label,
            "assessor": self.assessor,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkClaim:
    """A labeled source claim used to construct controlled responses.

    Benchmark claims are deliberately not policy views. Their assessment and
    native benchmark evidence remain on the evaluator/generator side of the
    boundary until a separate response-construction step freezes a response.
    """

    claim_id: str
    text: str
    provenance: Provenance
    assessment: GoldAssessment
    evidence: EvidenceBundle | None = None
    metadata: FrozenAttributes = field(default_factory=FrozenAttributes)

    def __post_init__(self) -> None:
        if not self.claim_id or not self.text.strip():
            raise ValueError("Benchmark claims require a non-empty ID and text")
        if self.assessment.unit_id != self.claim_id:
            raise ValueError("Benchmark claim and assessment IDs must match")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "provenance": self.provenance.to_dict(),
            "assessment": self.assessment.to_dict(),
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class PolicyVisibleUnit:
    """Leakage-minimized unit supplied to an ordinary policy.

    Native source identifiers, evaluator provenance, benchmark rationales and
    gold assessments intentionally remain outside this object. Evidence-aware
    policies will require an explicit, separately validated information view.
    """

    unit_id: str
    text: str
    observable_features: FrozenAttributes = field(default_factory=FrozenAttributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "observable_features": self.observable_features.to_dict(),
        }


@dataclass(frozen=True)
class PolicyView:
    """The only response representation passed to ordinary policies."""

    response_id: str
    units: tuple[PolicyVisibleUnit, ...]
    metadata: FrozenAttributes
    response_hash: str


@dataclass(frozen=True)
class EvidencePolicyVisibleUnit:
    """Gold-free unit view with only the cited evidence text needed by a policy."""

    unit_id: str
    text: str
    # Preserve native citation positions; None is a missing/blank text slot.
    evidence_texts: tuple[str | None, ...]
    missing_evidence_item_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "evidence_texts": list(self.evidence_texts),
            "missing_evidence_item_count": self.missing_evidence_item_count,
        }


@dataclass(frozen=True)
class EvidencePolicyView:
    """Explicit policy view for methods authorized to read cited evidence text."""

    response_id: str
    units: tuple[EvidencePolicyVisibleUnit, ...]
    metadata: FrozenAttributes
    response_hash: str


@dataclass(frozen=True)
class FrozenResponse:
    response_id: str
    units: tuple[VerificationUnit, ...]
    gold_assessments: tuple[GoldAssessment, ...]
    metadata: FrozenAttributes = field(default_factory=FrozenAttributes)

    def __post_init__(self) -> None:
        unit_ids = [unit.unit_id for unit in self.units]
        gold_ids = [assessment.unit_id for assessment in self.gold_assessments]
        if not self.response_id:
            raise ValueError("FrozenResponse requires a response_id")
        if not self.units:
            raise ValueError("FrozenResponse requires at least one unit")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Verification unit IDs must be unique")
        if len(gold_ids) != len(set(gold_ids)):
            raise ValueError("Gold assessment unit IDs must be unique")
        if set(unit_ids) != set(gold_ids):
            raise ValueError("Every unit requires exactly one gold assessment, including UNASSESSED")

    @property
    def response_hash(self) -> str:
        payload = {
            "response_id": self.response_id,
            "units": [unit.to_dict() for unit in self.units],
            "metadata": self.metadata.to_dict(),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def gold_hash(self) -> str:
        payload = [assessment.to_dict() for assessment in self.gold_assessments]
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def policy_view(self) -> PolicyView:
        units = tuple(
            PolicyVisibleUnit(
                unit_id=unit.unit_id,
                text=unit.text,
                observable_features=unit.observable_features,
            )
            for unit in self.units
        )
        visible_metadata = self.metadata.get("policy_visible", {})
        if not isinstance(visible_metadata, dict):
            raise ValueError("FrozenResponse metadata.policy_visible must be an object")
        payload = {
            "units": [unit.to_dict() for unit in units],
            "metadata": visible_metadata,
        }
        policy_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return PolicyView(
            response_id=f"policy-view:{policy_hash[:20]}",
            units=units,
            metadata=FrozenAttributes.from_mapping(visible_metadata),
            response_hash=policy_hash,
        )

    def assessment_map(self) -> dict[str, GoldAssessment]:
        return {assessment.unit_id: assessment for assessment in self.gold_assessments}

    def evidence_policy_view(self) -> EvidencePolicyView:
        units = []
        for unit in self.units:
            items = unit.evidence.items if unit.evidence else ()
            texts = tuple(
                item.text if isinstance(item.text, str) and item.text.strip() else None
                for item in items
            )
            units.append(EvidencePolicyVisibleUnit(
                unit_id=unit.unit_id,
                text=unit.text,
                evidence_texts=texts,
                missing_evidence_item_count=sum(not isinstance(item.text, str) or not item.text.strip() for item in items),
            ))
        visible_metadata = self.metadata.get("policy_visible", {})
        if not isinstance(visible_metadata, dict):
            raise ValueError("FrozenResponse metadata.policy_visible must be an object")
        payload = {"units": [unit.to_dict() for unit in units], "metadata": visible_metadata}
        policy_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return EvidencePolicyView(
            response_id=f"evidence-policy-view:{policy_hash[:20]}",
            units=tuple(units),
            metadata=FrozenAttributes.from_mapping(visible_metadata),
            response_hash=policy_hash,
        )
