"""Native FEVEROUS claim adapter.

The commonly distributed claim JSONL identifies Wikipedia evidence elements
but does not embed their text. ClaimReplay therefore retains element IDs and
contexts while keeping ``EvidenceItem.text`` optional.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..models import (
    BenchmarkClaim,
    EvidenceBundle,
    EvidenceItem,
    FrozenAttributes,
    GoldAssessment,
    Provenance,
    SupportLevel,
)
from .base import ClaimDatasetAdapter


def _normalise_label(label: Any) -> str:
    return " ".join(str(label).strip().upper().replace("_", " ").replace("-", " ").split())


FEVEROUS_SUPPORT_MAP = {
    "SUPPORTS": SupportLevel.FULL_SUPPORT,
    "REFUTES": SupportLevel.NO_SUPPORT,
    "NOT ENOUGH INFO": SupportLevel.NO_SUPPORT,
    "NEI": SupportLevel.NO_SUPPORT,
}


def _page_id(evidence_id: str) -> str:
    return re.split(r"_(?:sentence|cell|table|title|section|header_cell)_", evidence_id, maxsplit=1)[0]


class FeverousAdapter(ClaimDatasetAdapter):
    dataset_name = "feverous"

    def __init__(self, *, split: str = "unspecified") -> None:
        self.split = split

    def from_record(self, record: dict[str, Any]) -> BenchmarkClaim | None:
        raw_id = record.get("id")
        claim_text = str(record.get("claim") or "").strip()
        raw_label = str(record.get("label") or "").strip()
        # Some legacy exports contain a single all-empty header-like row.
        if not claim_text and not raw_label and raw_id in (None, ""):
            return None
        if raw_id in (None, "") or not claim_text or not raw_label:
            raise ValueError("FEVEROUS record requires id, claim, and label")

        normalised_label = _normalise_label(raw_label)
        try:
            support = FEVEROUS_SUPPORT_MAP[normalised_label]
        except KeyError as exc:
            raise ValueError(f"Unknown FEVEROUS label: {raw_label!r}") from exc

        evidence_sets = record.get("evidence") or []
        if not isinstance(evidence_sets, list):
            raise ValueError(f"FEVEROUS evidence for claim {raw_id!r} must be a list")
        citation_ids: list[str] = []
        evidence_occurrences: dict[str, list[dict[str, Any]]] = {}
        for set_index, evidence_set in enumerate(evidence_sets):
            if not isinstance(evidence_set, dict):
                raise ValueError(f"FEVEROUS evidence set for claim {raw_id!r} must be an object")
            contexts = evidence_set.get("context") or {}
            content = evidence_set.get("content") or []
            if not isinstance(content, list):
                raise ValueError(f"FEVEROUS evidence content for claim {raw_id!r} must be a list")
            for evidence_id in content:
                evidence_id = str(evidence_id)
                citation_ids.append(evidence_id)
                context = contexts.get(evidence_id, []) if isinstance(contexts, dict) else []
                evidence_occurrences.setdefault(evidence_id, []).append({
                    "evidence_set_index": set_index,
                    "context_ids": [str(value) for value in context],
                })
        items = [
            EvidenceItem(
                evidence_id=evidence_id,
                text=None,
                metadata=FrozenAttributes.from_mapping({
                    "occurrences": occurrences,
                }),
            )
            for evidence_id, occurrences in evidence_occurrences.items()
        ]

        claim_id = f"feverous:{raw_id}"
        provenance = Provenance(
            dataset=self.dataset_name,
            source_id=str(raw_id),
            citation_ids=tuple(dict.fromkeys(citation_ids)),
            metadata=FrozenAttributes.from_mapping({
                "split": self.split,
                "challenge": record.get("challenge"),
                "evidence_set_count": len(evidence_sets),
                "source_group_ids": sorted({_page_id(value) for value in citation_ids}),
            }),
        )
        return BenchmarkClaim(
            claim_id=claim_id,
            text=claim_text,
            provenance=provenance,
            assessment=GoldAssessment(
                unit_id=claim_id,
                support=support,
                raw_label=raw_label,
                assessor="FEVEROUS native annotation",
                metadata=FrozenAttributes.from_mapping({
                    "label_mapping": "SUPPORTS=>FULL; REFUTES/NEI=>NO",
                }),
            ),
            evidence=EvidenceBundle(tuple(items)) if items else None,
            metadata=FrozenAttributes.from_mapping({
                "word_count": len(claim_text.split()),
                "evidence_item_count": len(items),
                "native_evidence_text_available": False,
            }),
        )

    def load_claims(self, path: str | Path, *, limit: int | None = None) -> list[BenchmarkClaim]:
        output: list[BenchmarkClaim] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    claim = self.from_record(json.loads(line))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid FEVEROUS record at line {line_number}: {exc}") from exc
                if claim is not None:
                    output.append(claim)
                if limit is not None and len(output) >= limit:
                    break
        return output
