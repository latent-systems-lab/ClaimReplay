"""Ordinary and privileged verification-policy interfaces."""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import EvidencePolicyView, FrozenResponse, PolicyView
from .tasks import EvaluationTask


@dataclass(frozen=True)
class RankedUnit:
    unit_id: str
    score: float | None


@dataclass(frozen=True)
class TriageCost:
    """Observed ranking cost, distinct from the final inspection budget."""

    units_considered: int
    score_evaluations: int | None = None
    model_calls: int = 0
    retrieval_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    evidence_characters: int | None = None
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict:
        payload = {
            key: getattr(self, key)
            for key in (
                "units_considered", "score_evaluations", "model_calls", "retrieval_calls",
                "input_tokens", "output_tokens", "estimated_cost_usd",
            )
        }
        if self.evidence_characters is not None:
            payload["evidence_characters"] = self.evidence_characters
        if self.cache_hits or self.cache_misses:
            payload["cache_hits"] = self.cache_hits
            payload["cache_misses"] = self.cache_misses
        return payload


class VerificationPolicy(ABC):
    """Ordinary policies receive PolicyView, which has no gold assessments."""

    name: str

    @abstractmethod
    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        raise NotImplementedError

    def triage_cost(self, response: PolicyView) -> TriageCost:
        return TriageCost(units_considered=len(response.units))

    def observable_feature_schema(self) -> dict[str, str]:
        """Declare structured features required before ``rank`` is called.

        Supported public schema kinds are currently ``"number"`` and
        ``"string"``. Policies that use only text and opaque IDs return an
        empty schema. This is a validation contract, not a claim that
        innocently named features cannot contain deliberately disguised gold.
        """

        return {}


class EvidenceAwareVerificationPolicy(ABC):
    """Ordinary policy with explicit access to cited evidence text, never gold."""

    name: str

    @abstractmethod
    def rank(self, response: EvidencePolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        raise NotImplementedError

    def triage_cost(self, response: EvidencePolicyView) -> TriageCost:
        return TriageCost(units_considered=len(response.units))


class RandomPolicy(VerificationPolicy):
    name = "random"

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        unit_ids = [unit.unit_id for unit in response.units]
        random.Random(seed).shuffle(unit_ids)
        return tuple(RankedUnit(unit_id, None) for unit_id in unit_ids)

    def triage_cost(self, response: PolicyView) -> TriageCost:
        return TriageCost(units_considered=len(response.units), score_evaluations=0)


class FeatureScorePolicy(VerificationPolicy):
    def __init__(self, feature: str, *, descending: bool = True, name: str | None = None):
        self.feature = feature
        self.descending = descending
        self.name = name or f"feature:{feature}"

    def observable_feature_schema(self) -> dict[str, str]:
        return {self.feature: "number"}

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        scored = []
        for unit in response.units:
            value = unit.observable_features.get(self.feature)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Unit {unit.unit_id} lacks numeric feature {self.feature!r}")
            scored.append(RankedUnit(unit.unit_id, float(value)))
        return tuple(sorted(scored, key=lambda row: ((-row.score if self.descending else row.score), row.unit_id)))

    def triage_cost(self, response: PolicyView) -> TriageCost:
        return TriageCost(units_considered=len(response.units), score_evaluations=len(response.units))


class CitationScarcityPolicy(VerificationPolicy):
    """Prioritize units with fewer cited passages, using metadata only."""

    name = "baseline:citation-scarcity"

    def observable_feature_schema(self) -> dict[str, str]:
        return {"citation_count": "number"}

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        rows = []
        for unit in response.units:
            citation_count = unit.observable_features.get("citation_count")
            if not isinstance(citation_count, (int, float)) or isinstance(citation_count, bool):
                raise ValueError(f"Unit {unit.unit_id} lacks numeric feature 'citation_count'")
            rows.append(RankedUnit(unit.unit_id, -float(citation_count)))
        return tuple(sorted(rows, key=lambda row: (-float(row.score), row.unit_id)))

    def triage_cost(self, response: PolicyView) -> TriageCost:
        return TriageCost(units_considered=len(response.units), score_evaluations=len(response.units))


class LexicalCheckworthinessPolicy(VerificationPolicy):
    """Transparent claim-intrinsic baseline with no evidence or model calls.

    Score = token_count + 4 * numeric_token_count + 2 * capitalized_token_count.
    Capitalization in the sentence-initial token is ignored. This is a fixed,
    language-dependent heuristic baseline, not a learned verifier.
    """

    name = "baseline:lexical-checkworthiness-v1"
    _token_pattern = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

    @classmethod
    def score_text(cls, text: str) -> tuple[float, int]:
        tokens = cls._token_pattern.findall(text)
        numeric = sum(any(character.isdigit() for character in token) for token in tokens)
        capitalized = sum(token[:1].isupper() for token in tokens[1:])
        return float(len(tokens) + 4 * numeric + 2 * capitalized), len(tokens)

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        rows = [
            RankedUnit(unit.unit_id, self.score_text(unit.text)[0]) for unit in response.units
        ]
        return tuple(sorted(rows, key=lambda row: (-float(row.score), row.unit_id)))

    def triage_cost(self, response: PolicyView) -> TriageCost:
        token_count = sum(self.score_text(unit.text)[1] for unit in response.units)
        return TriageCost(
            units_considered=len(response.units),
            score_evaluations=len(response.units),
            input_tokens=token_count,
        )


class PrivilegedOraclePolicy:
    """Gold-aware sanity upper bound; intentionally not a VerificationPolicy."""

    name = "oracle"

    def rank(
        self, response: FrozenResponse, task: EvaluationTask, *, seed: int
    ) -> tuple[RankedUnit, ...]:
        del seed
        assessments = response.assessment_map()
        rows = [
            RankedUnit(unit.unit_id, float(assessments[unit.unit_id].support in task.positive_labels))
            for unit in response.units
        ]
        return tuple(sorted(rows, key=lambda row: (-float(row.score), row.unit_id)))

    def triage_cost(self, response: FrozenResponse) -> TriageCost:
        return TriageCost(units_considered=len(response.units), score_evaluations=len(response.units))
