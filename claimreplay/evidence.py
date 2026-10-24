"""Deterministic evidence cache loading and FrozenResponse materialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from .models import EvidenceBundle, EvidenceItem, FrozenResponse


@dataclass(frozen=True)
class EvidenceCacheEntry:
    evidence_id: str
    text: str
    text_sha256: str
    source: dict


@dataclass(frozen=True)
class EvidenceMaterializationReport:
    requested_item_count: int
    materialized_item_count: int
    already_materialized_item_count: int
    missing_item_count: int
    missing_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "requested_item_count": self.requested_item_count,
            "materialized_item_count": self.materialized_item_count,
            "already_materialized_item_count": self.already_materialized_item_count,
            "missing_item_count": self.missing_item_count,
            "missing_evidence_ids": list(self.missing_evidence_ids),
        }


class EvidenceTextCache:
    def __init__(self, entries: dict[str, EvidenceCacheEntry], *, source_sha256: str):
        self._entries = entries
        self.source_sha256 = source_sha256

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "EvidenceTextCache":
        path = Path(path)
        entries = {}
        raw = path.read_bytes()
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            evidence_id, text = str(row["evidence_id"]), row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty evidence text at line {line_number}")
            expected_hash = row.get("source", {}).get("text_sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()
            actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if expected_hash != actual_hash:
                raise ValueError(f"Evidence text hash mismatch for {evidence_id}")
            if evidence_id in entries:
                raise ValueError(f"Duplicate evidence ID {evidence_id}")
            entries[evidence_id] = EvidenceCacheEntry(evidence_id, text, actual_hash, dict(row.get("source", {})))
        return cls(entries, source_sha256=hashlib.sha256(raw).hexdigest())

    def get(self, evidence_id: str) -> EvidenceCacheEntry | None:
        return self._entries.get(evidence_id)

    def __len__(self) -> int:
        return len(self._entries)


def materialize_response_evidence(
    response: FrozenResponse, cache: EvidenceTextCache
) -> tuple[FrozenResponse, EvidenceMaterializationReport]:
    units = []
    requested = materialized = already = 0
    missing = []
    for unit in response.units:
        if unit.evidence is None:
            units.append(unit)
            continue
        items = []
        for item in unit.evidence.items:
            requested += 1
            cached = cache.get(item.evidence_id)
            if isinstance(item.text, str) and item.text.strip():
                if cached is not None and item.text != cached.text:
                    raise ValueError(f"Existing evidence conflicts with cache for {item.evidence_id}")
                items.append(item)
                already += 1
            elif cached is None:
                items.append(item)
                missing.append(item.evidence_id)
            else:
                items.append(replace(item, text=cached.text))
                materialized += 1
        units.append(replace(unit, evidence=EvidenceBundle(tuple(items))))
    updated = replace(response, units=tuple(units))
    updated_metadata = updated.metadata.to_dict()
    updated_metadata["evidence_cache_sha256"] = cache.source_sha256
    updated = replace(updated, metadata=type(updated.metadata).from_mapping(updated_metadata))
    report = EvidenceMaterializationReport(
        requested_item_count=requested,
        materialized_item_count=materialized,
        already_materialized_item_count=already,
        missing_item_count=len(missing),
        missing_evidence_ids=tuple(sorted(set(missing))),
    )
    return updated, report
