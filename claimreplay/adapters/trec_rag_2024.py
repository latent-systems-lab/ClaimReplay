"""Adapter for official TREC RAG 2024 citation/support judgments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..models import (
    EvidenceBundle,
    EvidenceItem,
    FrozenAttributes,
    FrozenResponse,
    GoldAssessment,
    Provenance,
    SupportLevel,
    VerificationUnit,
)
from .base import DatasetAdapter


SUPPORT_MAP = {
    "0": SupportLevel.NO_SUPPORT,
    "1": SupportLevel.PARTIAL_SUPPORT,
    "2": SupportLevel.FULL_SUPPORT,
}


class TrecRag2024Adapter(DatasetAdapter):
    """Map one official JSONL row to one frozen response.

    TREC's official support evaluation uses the first/highest-ranked citation.
    All citation IDs are retained as provenance, while the first citation's
    assessment supplies the canonical support level. Raw passage text is
    optional and is not required by the ClaimReplay core.
    """

    dataset_name = "trec_rag_2024"

    def __init__(self, *, judgment_condition: str = "unspecified") -> None:
        self.judgment_condition = judgment_condition

    def from_record(self, record: dict) -> FrozenResponse:
        topic_id = str(record["topic_id"])
        run_id = str(record["run_id"])
        units = []
        assessments = []
        skipped_empty_sentences = 0
        for index, sentence in enumerate(record.get("sentences", [])):
            text = str(sentence.get("text", "")).strip()
            if not text:
                skipped_empty_sentences += 1
                continue
            sentence_id = sentence.get("sentenceID", sentence.get("sentenceId", index))
            unit_id = f"s{int(sentence_id):04d}"
            citations = sentence.get("citations") or []
            citation_ids = tuple(str(row.get("reference")) for row in citations if row.get("reference"))
            evidence = EvidenceBundle(tuple(
                EvidenceItem(
                    evidence_id=str(row.get("reference") or f"citation:{position}"),
                    text=row.get("text"),
                    metadata=FrozenAttributes.from_mapping({"citation_position": position}),
                )
                for position, row in enumerate(citations)
            )) if citations else None
            raw_support = citations[0].get("support") if citations else None
            first_support = str(raw_support) if raw_support is not None else None
            support = SUPPORT_MAP.get(first_support, SupportLevel.UNASSESSED)
            units.append(VerificationUnit(
                unit_id=unit_id,
                text=text,
                provenance=Provenance(
                    dataset=self.dataset_name,
                    source_id=f"{topic_id}:{run_id}:{sentence_id}",
                    citation_ids=citation_ids,
                    metadata=FrozenAttributes.from_mapping({
                        "topic_id": topic_id,
                        "run_id": run_id,
                        "sentence_id": sentence_id,
                        "citation_count": len(citations),
                        "judgment_condition": self.judgment_condition,
                    }),
                ),
                observable_features=FrozenAttributes.from_mapping({
                    "citation_count": len(citations),
                }),
                evidence=evidence,
            ))
            assessments.append(GoldAssessment(
                unit_id=unit_id,
                support=support,
                raw_label=first_support,
                assessor="TREC RAG 2024 human assessment",
                metadata=FrozenAttributes.from_mapping({
                    "mapping": "first_citation_support",
                    "has_citation": bool(citations),
                    "uncited_is_unassessed": not citations,
                    "verification_worthy": sentence.get("verification_worthy"),
                }),
            ))
        if not units:
            raise ValueError(f"TREC response {topic_id}:{run_id} has no non-empty sentence units")
        return FrozenResponse(
            response_id=f"trec-rag-2024:{topic_id}:{run_id}",
            units=tuple(units),
            gold_assessments=tuple(assessments),
            metadata=FrozenAttributes.from_mapping({
                "dataset": self.dataset_name,
                "topic_id": topic_id,
                "run_id": run_id,
                "judgment_condition": self.judgment_condition,
                "skipped_empty_sentence_count": skipped_empty_sentences,
                "support_mapping": {"0": "NO_SUPPORT", "1": "PARTIAL_SUPPORT", "2": "FULL_SUPPORT"},
                "unassessed_mapping": "uncited, missing, or unknown/-1 first-citation support",
                "policy_visible": {"response_kind": "naturalistic_trec_rag_answer"},
            }),
        )

    def load_jsonl(self, path: str | Path, *, limit: int | None = None) -> list[FrozenResponse]:
        output = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                output.append(self.from_record(json.loads(line)))
                if limit is not None and len(output) >= limit:
                    break
        return output

    def load(self, path: str | Path, *, limit: int | None = None) -> list[FrozenResponse]:
        return self.load_jsonl(path, limit=limit)
