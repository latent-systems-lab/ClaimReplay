"""Provider-neutral controlled realization with cache and validation gates."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .controlled import ControlledInstanceManifest
from .models import (
    FrozenAttributes,
    FrozenResponse,
    GoldAssessment,
    Provenance,
    VerificationUnit,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GenerationItem:
    item_id: str
    source_text: str

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    provider: str
    model: str
    prompt_version: str
    items: tuple[GenerationItem, ...]
    parameters: FrozenAttributes = field(default_factory=FrozenAttributes)

    def __post_init__(self) -> None:
        if not self.request_id or not self.provider or not self.model or not self.prompt_version:
            raise ValueError("Generation request identifiers must be non-empty")
        if not self.items:
            raise ValueError("Generation request requires at least one item")
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Generation request item IDs must be unique")

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "items": [item.to_dict() for item in self.items],
            "parameters": self.parameters.to_dict(),
        }

    @property
    def request_hash(self) -> str:
        return _hash(self.to_dict())

    @property
    def cache_key(self) -> str:
        # The request contains no labels, prevalence, assessments or gold rationale.
        return self.request_hash


def request_from_manifest(
    manifest: ControlledInstanceManifest,
    *,
    provider: str,
    model: str,
    parameters: dict[str, Any] | None = None,
) -> GenerationRequest:
    items = tuple(GenerationItem(**item) for item in manifest.provider_items())
    identity = {
        "provider": provider,
        "model": model,
        "prompt_version": manifest.config.prompt_version,
        "items": [item.to_dict() for item in items],
        "parameters": parameters or {},
    }
    return GenerationRequest(
        request_id=f"generation:{_hash(identity)[:24]}",
        provider=provider,
        model=model,
        prompt_version=manifest.config.prompt_version,
        items=items,
        parameters=FrozenAttributes.from_mapping(parameters),
    )


@dataclass(frozen=True)
class GeneratedUnit:
    item_id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GenerationUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_characters: int | None = None
    output_characters: int | None = None
    estimated_cost_usd: float | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ProviderResponse:
    response_id: str
    request_hash: str
    provider: str
    model: str
    units: tuple[GeneratedUnit, ...]
    usage: GenerationUsage = field(default_factory=GenerationUsage)
    provider_metadata: FrozenAttributes = field(default_factory=FrozenAttributes)

    def to_dict(self) -> dict:
        return {
            "response_id": self.response_id,
            "request_hash": self.request_hash,
            "provider": self.provider,
            "model": self.model,
            "units": [unit.to_dict() for unit in self.units],
            "usage": self.usage.to_dict(),
            "provider_metadata": self.provider_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProviderResponse":
        return cls(
            response_id=payload["response_id"],
            request_hash=payload["request_hash"],
            provider=payload["provider"],
            model=payload["model"],
            units=tuple(GeneratedUnit(**unit) for unit in payload["units"]),
            usage=GenerationUsage(**payload.get("usage", {})),
            provider_metadata=FrozenAttributes.from_mapping(payload.get("provider_metadata")),
        )


class GenerationProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, request: GenerationRequest, *, attempt: int) -> ProviderResponse:
        raise NotImplementedError


class MockGenerationProvider(GenerationProvider):
    """Offline deterministic provider; exact mode is semantically lossless."""

    name = "mock"

    def __init__(self, *, mode: str = "exact") -> None:
        if mode not in {"exact", "prefix", "empty"}:
            raise ValueError("Mock mode must be exact, prefix, or empty")
        self.mode = mode
        self.call_count = 0

    def generate(self, request: GenerationRequest, *, attempt: int) -> ProviderResponse:
        self.call_count += 1
        texts = []
        for item in request.items:
            if self.mode == "exact":
                text = item.source_text
            elif self.mode == "prefix":
                text = f"Reportedly, {item.source_text}"
            else:
                text = ""
            texts.append(GeneratedUnit(item.item_id, text))
        output_characters = sum(len(unit.text) for unit in texts)
        return ProviderResponse(
            response_id=f"mock:{request.request_hash[:20]}:{attempt}",
            request_hash=request.request_hash,
            provider=self.name,
            model=request.model,
            units=tuple(texts),
            usage=GenerationUsage(
                input_characters=sum(len(item.source_text) for item in request.items),
                output_characters=output_characters,
                estimated_cost_usd=0.0,
            ),
            provider_metadata=FrozenAttributes.from_mapping({
                "mode": self.mode,
                "attempt": attempt,
                "deterministic": True,
            }),
        )


class SemanticStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class SemanticValidation:
    item_id: str
    source_text: str
    generated_text: str
    status: SemanticStatus
    validator: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "status": self.status.value,
        }


class SemanticValidator(ABC):
    name: str

    @abstractmethod
    def validate(self, item: GenerationItem, generated: GeneratedUnit) -> SemanticValidation:
        raise NotImplementedError


class ExactTextSemanticValidator(SemanticValidator):
    """Certifies exact normalized copies; altered text is sent to review."""

    name = "exact-text-v1"

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    def validate(self, item: GenerationItem, generated: GeneratedUnit) -> SemanticValidation:
        source = self._normalize(item.source_text)
        realization = self._normalize(generated.text)
        if not realization:
            status = SemanticStatus.REJECTED
            reason = "empty generated realization"
        elif source == realization:
            status = SemanticStatus.ACCEPTED
            reason = None
        else:
            status = SemanticStatus.NEEDS_REVIEW
            reason = "non-exact realization; semantic faithfulness not automatically established"
        return SemanticValidation(
            item_id=item.item_id,
            source_text=item.source_text,
            generated_text=generated.text,
            status=status,
            validator=self.name,
            reason=reason,
        )


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retry_schema_errors: bool = True
    retry_semantic_rejections: bool = True
    return_needs_review: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")


@dataclass(frozen=True)
class GenerationArtifact:
    request: GenerationRequest
    response: ProviderResponse | None
    validations: tuple[SemanticValidation, ...]
    status: SemanticStatus
    attempts: int
    from_cache: bool
    exclusion_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "request": self.request.to_dict(),
            "request_hash": self.request.request_hash,
            "response": self.response.to_dict() if self.response else None,
            "validations": [validation.to_dict() for validation in self.validations],
            "status": self.status.value,
            "attempts": self.attempts,
            "from_cache": self.from_cache,
            "exclusion_reason": self.exclusion_reason,
        }


class ReviewDecisionValue(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class SemanticReviewDecision:
    item_id: str
    decision: ReviewDecisionValue
    reviewer: str
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "decision": self.decision.value,
            "reviewer": self.reviewer,
            "notes": self.notes,
        }


class GenerationCache:
    """File-per-request cache whose keys are label-independent request hashes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ProviderResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        return ProviderResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def put(self, key: str, response: ProviderResponse) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


def _schema_errors(request: GenerationRequest, response: ProviderResponse) -> list[str]:
    errors: list[str] = []
    if response.request_hash != request.request_hash:
        errors.append("provider response request_hash mismatch")
    if response.provider != request.provider or response.model != request.model:
        errors.append("provider/model provenance mismatch")
    actual = [unit.item_id for unit in response.units]
    expected = [item.item_id for item in request.items]
    if len(actual) != len(set(actual)):
        errors.append("duplicate generated item IDs")
    if set(actual) != set(expected):
        errors.append("generated item IDs do not exactly match request")
    if any(not unit.text.strip() for unit in response.units):
        errors.append("one or more generated realizations are empty")
    return errors


class GenerationRunner:
    def __init__(
        self,
        provider: GenerationProvider,
        validator: SemanticValidator,
        *,
        cache: GenerationCache | None = None,
        retry_policy: RetryPolicy = RetryPolicy(),
    ) -> None:
        self.provider = provider
        self.validator = validator
        self.cache = cache
        self.retry_policy = retry_policy

    def run(self, request: GenerationRequest) -> GenerationArtifact:
        if request.provider != self.provider.name:
            raise ValueError("Request provider does not match GenerationProvider")
        cached = self.cache.get(request.cache_key) if self.cache else None
        if cached is not None:
            return self._validate(request, cached, attempts=0, from_cache=True)

        last_response: ProviderResponse | None = None
        last_reason: str | None = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            response = self.provider.generate(request, attempt=attempt)
            last_response = response
            errors = _schema_errors(request, response)
            if errors:
                last_reason = "; ".join(errors)
                if self.retry_policy.retry_schema_errors and attempt < self.retry_policy.max_attempts:
                    continue
                return GenerationArtifact(
                    request=request,
                    response=response,
                    validations=(),
                    status=SemanticStatus.REJECTED,
                    attempts=attempt,
                    from_cache=False,
                    exclusion_reason=last_reason,
                )
            artifact = self._validate(request, response, attempts=attempt, from_cache=False)
            if artifact.status is SemanticStatus.REJECTED:
                last_reason = artifact.exclusion_reason
                if self.retry_policy.retry_semantic_rejections and attempt < self.retry_policy.max_attempts:
                    continue
            if artifact.status is SemanticStatus.NEEDS_REVIEW and not self.retry_policy.return_needs_review:
                return GenerationArtifact(
                    request=request,
                    response=response,
                    validations=artifact.validations,
                    status=SemanticStatus.REJECTED,
                    attempts=attempt,
                    from_cache=False,
                    exclusion_reason="semantic review required but pending outputs are disallowed",
                )
            if self.cache and artifact.status is not SemanticStatus.REJECTED:
                self.cache.put(request.cache_key, response)
            return artifact
        raise RuntimeError(f"Generation exhausted retries: {last_reason}; response={last_response}")

    def _validate(
        self,
        request: GenerationRequest,
        response: ProviderResponse,
        *,
        attempts: int,
        from_cache: bool,
    ) -> GenerationArtifact:
        errors = _schema_errors(request, response)
        if errors:
            return GenerationArtifact(
                request, response, (), SemanticStatus.REJECTED, attempts, from_cache, "; ".join(errors)
            )
        by_id = {unit.item_id: unit for unit in response.units}
        validations = tuple(
            self.validator.validate(item, by_id[item.item_id]) for item in request.items
        )
        statuses = {validation.status for validation in validations}
        if SemanticStatus.REJECTED in statuses:
            status = SemanticStatus.REJECTED
            reason = "semantic validation rejected one or more realizations"
        elif SemanticStatus.NEEDS_REVIEW in statuses:
            status = SemanticStatus.NEEDS_REVIEW
            reason = "semantic faithfulness requires explicit review"
        else:
            status = SemanticStatus.ACCEPTED
            reason = None
        return GenerationArtifact(
            request, response, validations, status, attempts, from_cache, reason
        )


def freeze_controlled_response(
    manifest: ControlledInstanceManifest,
    artifact: GenerationArtifact,
) -> FrozenResponse:
    """Freeze only semantically accepted realizations with evaluator-only gold."""

    if artifact.status is not SemanticStatus.ACCEPTED or artifact.response is None:
        raise ValueError("Only ACCEPTED generation artifacts can be frozen for evaluation")
    expected = {entry.unit_id for entry in manifest.entries}
    actual = {unit.item_id for unit in artifact.response.units}
    if expected != actual:
        raise ValueError("Manifest and generated realization IDs do not match")
    generated = {unit.item_id: unit for unit in artifact.response.units}
    validations = {row.item_id: row for row in artifact.validations}
    units = []
    assessments = []
    for entry in manifest.entries:
        claim = entry.claim
        units.append(VerificationUnit(
            unit_id=entry.unit_id,
            text=generated[entry.unit_id].text,
            provenance=Provenance(
                dataset=claim.provenance.dataset,
                source_id=claim.provenance.source_id,
                citation_ids=claim.provenance.citation_ids,
                metadata=FrozenAttributes.from_mapping({
                    **claim.provenance.metadata.to_dict(),
                    "benchmark_claim_id": claim.claim_id,
                    "generation_request_hash": artifact.request.request_hash,
                    "semantic_validation": validations[entry.unit_id].status.value,
                }),
            ),
            observable_features=FrozenAttributes.from_mapping({
                "position": entry.position,
                "word_count": len(generated[entry.unit_id].text.split()),
            }),
            # Native gold rationales are evaluator-side in controlled bundles.
            evidence=None,
        ))
        assessments.append(GoldAssessment(
            unit_id=entry.unit_id,
            support=claim.assessment.support,
            raw_label=claim.assessment.raw_label,
            assessor=claim.assessment.assessor,
            metadata=FrozenAttributes.from_mapping({
                **claim.assessment.metadata.to_dict(),
                "benchmark_claim_id": claim.claim_id,
            }),
        ))
    return FrozenResponse(
        response_id=f"controlled-response:{manifest.manifest_hash[:24]}",
        units=tuple(units),
        gold_assessments=tuple(assessments),
        metadata=FrozenAttributes.from_mapping({
            "response_kind": "controlled_claim_bundle",
            "not_naturalistic_rag_answer": True,
            "manifest_id": manifest.manifest_id,
            "manifest_hash": manifest.manifest_hash,
            "group_id": manifest.config.group_id,
            "generation_request_hash": artifact.request.request_hash,
            "provider": artifact.response.provider,
            "model": artifact.response.model,
            "policy_visible": {"response_kind": "controlled_claim_bundle"},
        }),
    )


def export_semantic_review_queue(artifact: GenerationArtifact, path: str | Path) -> int:
    """Export altered realizations for label-blind semantic review."""

    pending = [row for row in artifact.validations if row.status is SemanticStatus.NEEDS_REVIEW]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps({
                "request_id": artifact.request.request_id,
                "item_id": row.item_id,
                "source_text": row.source_text,
                "generated_text": row.generated_text,
                "automatic_status": row.status.value,
                "automatic_reason": row.reason,
                "decision": None,
                "reviewer": None,
                "notes": None,
            }, ensure_ascii=False, sort_keys=True) + "\n"
            for row in pending
        ),
        encoding="utf-8",
    )
    return len(pending)


def adjudicate_generation_artifact(
    artifact: GenerationArtifact,
    decisions: Iterable[SemanticReviewDecision],
) -> GenerationArtifact:
    """Apply explicit human decisions; never auto-accept an altered realization."""

    if artifact.status is not SemanticStatus.NEEDS_REVIEW:
        raise ValueError("Only NEEDS_REVIEW artifacts can be adjudicated")
    decision_map = {decision.item_id: decision for decision in decisions}
    pending_ids = {
        row.item_id for row in artifact.validations if row.status is SemanticStatus.NEEDS_REVIEW
    }
    if set(decision_map) != pending_ids:
        missing = sorted(pending_ids - set(decision_map))
        extra = sorted(set(decision_map) - pending_ids)
        raise ValueError(f"Review decisions must exactly cover pending items; missing={missing}, extra={extra}")
    revised = []
    rejected = False
    for row in artifact.validations:
        if row.status is not SemanticStatus.NEEDS_REVIEW:
            revised.append(row)
            continue
        decision = decision_map[row.item_id]
        if not decision.reviewer.strip():
            raise ValueError("Semantic review decisions require a reviewer identifier")
        accepted = decision.decision is ReviewDecisionValue.ACCEPT
        rejected = rejected or not accepted
        revised.append(SemanticValidation(
            item_id=row.item_id,
            source_text=row.source_text,
            generated_text=row.generated_text,
            status=SemanticStatus.ACCEPTED if accepted else SemanticStatus.REJECTED,
            validator=f"human-review:{decision.reviewer}",
            reason=decision.notes,
        ))
    return GenerationArtifact(
        request=artifact.request,
        response=artifact.response,
        validations=tuple(revised),
        status=SemanticStatus.REJECTED if rejected else SemanticStatus.ACCEPTED,
        attempts=artifact.attempts,
        from_cache=artifact.from_cache,
        exclusion_reason="one or more semantic reviews rejected the realization" if rejected else None,
    )


def save_generation_artifact(artifact: GenerationArtifact, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
