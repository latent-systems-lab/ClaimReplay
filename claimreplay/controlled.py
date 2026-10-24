"""Deterministic construction of controlled benchmark claim bundles."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from .models import BenchmarkClaim, SupportLevel
from .tasks import EvaluationTask


SCHEMA_VERSION = "claimreplay.controlled-manifest.v1"
ALLOWED_RESPONSE_SIZES = frozenset({20, 40})
ALLOWED_PREVALENCES = frozenset({Decimal("0.05"), Decimal("0.10"), Decimal("0.20")})


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_groups(claim: BenchmarkClaim) -> tuple[str, ...]:
    declared = claim.provenance.metadata.get("source_group_ids")
    if isinstance(declared, list) and declared:
        return tuple(sorted({str(value) for value in declared}))
    if claim.provenance.citation_ids:
        return tuple(sorted(set(claim.provenance.citation_ids)))
    return (f"{claim.provenance.dataset}:{claim.provenance.source_id or claim.claim_id}",)


@dataclass(frozen=True)
class ControlledResponseConfig:
    group_id: str
    response_size: int
    positive_prevalence: Decimal | float | str
    seed: int
    split: str
    prompt_version: str = "exact-copy-v1"
    disjoint_sources_within_group: bool = True

    def __post_init__(self) -> None:
        prevalence = Decimal(str(self.positive_prevalence))
        object.__setattr__(self, "positive_prevalence", prevalence)
        if not self.group_id:
            raise ValueError("Controlled configuration requires a group_id")
        if self.response_size not in ALLOWED_RESPONSE_SIZES:
            raise ValueError(f"response_size must be one of {sorted(ALLOWED_RESPONSE_SIZES)}")
        if prevalence not in ALLOWED_PREVALENCES:
            raise ValueError("positive_prevalence must be one of 0.05, 0.10, 0.20")
        if not self.split:
            raise ValueError("Controlled configuration requires an explicit split")

    @property
    def positive_count(self) -> int:
        count = Decimal(self.response_size) * self.positive_prevalence
        if count != count.to_integral_value():
            raise ValueError(
                f"Prevalence {self.positive_prevalence} is not an exact integer count at n={self.response_size}"
            )
        return int(count)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "response_size": self.response_size,
            "positive_prevalence": format(self.positive_prevalence, "f"),
            "positive_count": self.positive_count,
            "seed": self.seed,
            "split": self.split,
            "prompt_version": self.prompt_version,
            "disjoint_sources_within_group": self.disjoint_sources_within_group,
        }


@dataclass(frozen=True)
class ManifestEntry:
    position: int
    unit_id: str
    claim: BenchmarkClaim
    source_group_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "unit_id": self.unit_id,
            "source_group_ids": list(self.source_group_ids),
            "benchmark_claim": self.claim.to_dict(),
        }

    def provider_item(self) -> dict[str, str]:
        """Return only generation-visible fields; no label or gold provenance."""

        return {
            "item_id": self.unit_id,
            "source_text": self.claim.text,
        }


@dataclass(frozen=True)
class ControlledInstanceManifest:
    manifest_id: str
    dataset: str
    task_name: str
    config: ControlledResponseConfig
    entries: tuple[ManifestEntry, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "dataset": self.dataset,
            "task_name": self.task_name,
            "config": self.config.to_dict(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def manifest_hash(self) -> str:
        payload = self.to_dict().copy()
        payload.pop("manifest_id", None)
        return _sha256(payload)

    def provider_items(self) -> tuple[dict[str, str], ...]:
        return tuple(entry.provider_item() for entry in self.entries)


class ControlledResponseBuilder:
    """Evaluator-side, label-aware sampler for controlled claim bundles."""

    def __init__(self, claims: Iterable[BenchmarkClaim], task: EvaluationTask):
        self.claims = tuple(claims)
        self.task = task
        if not self.claims:
            raise ValueError("ControlledResponseBuilder requires at least one benchmark claim")
        datasets = {claim.provenance.dataset for claim in self.claims}
        if len(datasets) != 1:
            raise ValueError("One controlled builder cannot silently mix benchmark datasets")
        self.dataset = next(iter(datasets))
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("Benchmark claim IDs must be unique")

    @staticmethod
    def _pick(
        candidates: list[BenchmarkClaim],
        count: int,
        *,
        rng: random.Random,
        blocked_claims: set[str],
        blocked_sources: set[str],
        disjoint_within: bool,
        label_role: str,
    ) -> list[BenchmarkClaim]:
        ordered = sorted(candidates, key=lambda claim: claim.claim_id)
        rng.shuffle(ordered)
        selected: list[BenchmarkClaim] = []
        local_sources: set[str] = set()
        for claim in ordered:
            groups = set(_source_groups(claim))
            if claim.claim_id in blocked_claims or groups & blocked_sources:
                continue
            if disjoint_within and groups & local_sources:
                continue
            selected.append(claim)
            local_sources.update(groups)
            if len(selected) == count:
                return selected
        raise ValueError(
            f"Infeasible controlled configuration: requested {count} {label_role} claims, "
            f"but only {len(selected)} satisfy claim/source-disjointness constraints"
        )

    def build(
        self,
        config: ControlledResponseConfig,
        *,
        reserved_claim_ids: set[str] | None = None,
        reserved_source_group_ids: set[str] | None = None,
    ) -> ControlledInstanceManifest:
        reserved_claims = set(reserved_claim_ids or ())
        reserved_sources = set(reserved_source_group_ids or ())
        eligible = [
            claim for claim in self.claims
            if claim.assessment.support in self.task.eligible_labels
        ]
        positive = [
            claim for claim in eligible
            if claim.assessment.support in self.task.positive_labels
        ]
        negative = [
            claim for claim in eligible
            if claim.assessment.support not in self.task.positive_labels
        ]
        rng = random.Random(config.seed)
        picked_positive = self._pick(
            positive,
            config.positive_count,
            rng=rng,
            blocked_claims=reserved_claims,
            blocked_sources=reserved_sources,
            disjoint_within=config.disjoint_sources_within_group,
            label_role="task-positive",
        )
        positive_sources = {group for claim in picked_positive for group in _source_groups(claim)}
        picked_negative = self._pick(
            negative,
            config.response_size - config.positive_count,
            rng=rng,
            blocked_claims=reserved_claims | {claim.claim_id for claim in picked_positive},
            blocked_sources=reserved_sources | positive_sources,
            disjoint_within=config.disjoint_sources_within_group,
            label_role="task-negative",
        )
        selected = picked_positive + picked_negative
        rng.shuffle(selected)
        entries = tuple(
            ManifestEntry(
                position=index,
                unit_id=f"u{index + 1:04d}",
                claim=claim,
                source_group_ids=_source_groups(claim),
            )
            for index, claim in enumerate(selected)
        )
        identity = {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "task_name": self.task.name,
            "config": config.to_dict(),
            "claim_ids": [entry.claim.claim_id for entry in entries],
        }
        return ControlledInstanceManifest(
            manifest_id=f"controlled:{_sha256(identity)[:24]}",
            dataset=self.dataset,
            task_name=self.task.name,
            config=config,
            entries=entries,
        )

    def build_many(self, configs: Iterable[ControlledResponseConfig]) -> tuple[ControlledInstanceManifest, ...]:
        manifests: list[ControlledInstanceManifest] = []
        used_claims: set[str] = set()
        used_sources: set[str] = set()
        seen_groups: set[str] = set()
        for config in configs:
            if config.group_id in seen_groups:
                raise ValueError(f"Duplicate controlled group_id: {config.group_id}")
            manifest = self.build(
                config,
                reserved_claim_ids=used_claims,
                reserved_source_group_ids=used_sources,
            )
            manifests.append(manifest)
            seen_groups.add(config.group_id)
            used_claims.update(entry.claim.claim_id for entry in manifest.entries)
            used_sources.update(group for entry in manifest.entries for group in entry.source_group_ids)
        return tuple(manifests)


def save_manifest(manifest: ControlledInstanceManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
