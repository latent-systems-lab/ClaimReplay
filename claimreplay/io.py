"""JSON serialization for the public canonical response format."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import (
    EvidenceBundle,
    EvidenceItem,
    FrozenAttributes,
    FrozenResponse,
    GoldAssessment,
    Provenance,
    SupportLevel,
    VerificationUnit,
)
from .replay import ReplayTrace


def response_from_dict(payload: dict) -> FrozenResponse:
    units = []
    for row in payload["units"]:
        provenance = row["provenance"]
        evidence_payload = row.get("evidence")
        evidence = None
        if evidence_payload is not None:
            evidence = EvidenceBundle(tuple(
                EvidenceItem(
                    evidence_id=item["evidence_id"],
                    text=item.get("text"),
                    source_uri=item.get("source_uri"),
                    metadata=FrozenAttributes.from_mapping(item.get("metadata")),
                )
                for item in evidence_payload.get("items", [])
            ))
        units.append(VerificationUnit(
            unit_id=row["unit_id"],
            text=row["text"],
            provenance=Provenance(
                dataset=provenance["dataset"],
                source_id=provenance.get("source_id"),
                citation_ids=tuple(provenance.get("citation_ids", [])),
                metadata=FrozenAttributes.from_mapping(provenance.get("metadata")),
            ),
            observable_features=FrozenAttributes.from_mapping(row.get("observable_features")),
            evidence=evidence,
        ))
    assessments = tuple(
        GoldAssessment(
            unit_id=row["unit_id"],
            support=SupportLevel(row["support"]),
            raw_label=row.get("raw_label"),
            assessor=row.get("assessor"),
            metadata=FrozenAttributes.from_mapping(row.get("metadata")),
        )
        for row in payload["gold_assessments"]
    )
    return FrozenResponse(
        response_id=payload["response_id"],
        units=tuple(units),
        gold_assessments=assessments,
        metadata=FrozenAttributes.from_mapping(payload.get("metadata")),
    )


def load_response(path: str | Path) -> FrozenResponse:
    return response_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def response_to_dict(response: FrozenResponse) -> dict:
    return {
        "response_id": response.response_id,
        "units": [unit.to_dict() for unit in response.units],
        "gold_assessments": [row.to_dict() for row in response.gold_assessments],
        "metadata": response.metadata.to_dict(),
    }


def save_response(response: FrozenResponse, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(response_to_dict(response), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_trace(trace: ReplayTrace, path: str | Path) -> None:
    """Atomically export a machine-readable replay trace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
