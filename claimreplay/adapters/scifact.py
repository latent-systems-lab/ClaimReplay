"""Native SciFact claims/corpus adapter with sentence-level evidence joins."""

from __future__ import annotations

import json
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


SCIFACT_SUPPORT_MAP = {
    "SUPPORT": SupportLevel.FULL_SUPPORT,
    "CONTRADICT": SupportLevel.NO_SUPPORT,
    "NOT_ENOUGH_INFO": SupportLevel.NO_SUPPORT,
}


class SciFactAdapter(ClaimDatasetAdapter):
    dataset_name = "scifact"

    def __init__(self, corpus_path: str | Path, *, split: str = "unspecified") -> None:
        self.corpus_path = Path(corpus_path)
        self.split = split
        self._corpus: dict[str, dict[str, Any]] | None = None

    def _load_corpus(self) -> dict[str, dict[str, Any]]:
        if self._corpus is not None:
            return self._corpus
        corpus: dict[str, dict[str, Any]] = {}
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_id = str(record.get("doc_id", ""))
                if not doc_id:
                    raise ValueError(f"SciFact corpus record at line {line_number} has no doc_id")
                if doc_id in corpus:
                    raise ValueError(f"Duplicate SciFact corpus doc_id: {doc_id}")
                corpus[doc_id] = record
        self._corpus = corpus
        return corpus

    @staticmethod
    def _rationale_text(document: dict[str, Any], sentence_ids: list[int], *, claim_id: str) -> str:
        abstract = document.get("abstract") or []
        sentences: list[str] = []
        for sentence_id in sentence_ids:
            if not isinstance(sentence_id, int) or sentence_id < 0 or sentence_id >= len(abstract):
                raise ValueError(
                    f"SciFact claim {claim_id} has out-of-range sentence {sentence_id} "
                    f"for document {document.get('doc_id')}"
                )
            sentences.append(str(abstract[sentence_id]))
        return " ".join(sentences).strip()

    def from_record(self, record: dict[str, Any]) -> BenchmarkClaim:
        corpus = self._load_corpus()
        raw_id = record.get("id")
        claim_text = str(record.get("claim") or "").strip()
        if raw_id in (None, "") or not claim_text:
            raise ValueError("SciFact record requires id and claim")
        native_evidence = record.get("evidence") or {}
        if not isinstance(native_evidence, dict):
            raise ValueError(f"SciFact evidence for claim {raw_id!r} must be an object")

        labels: set[str] = set()
        items: list[EvidenceItem] = []
        for doc_id_raw, rationales in native_evidence.items():
            doc_id = str(doc_id_raw)
            document = corpus.get(doc_id)
            if document is None:
                raise ValueError(f"SciFact claim {raw_id} references missing corpus document {doc_id}")
            for rationale_index, rationale in enumerate(rationales):
                label = str(rationale.get("label") or "").strip().upper()
                if label not in {"SUPPORT", "CONTRADICT"}:
                    raise ValueError(f"Unknown SciFact rationale label: {label!r}")
                labels.add(label)
                sentence_ids = rationale.get("sentences") or []
                text = self._rationale_text(document, sentence_ids, claim_id=str(raw_id))
                items.append(EvidenceItem(
                    evidence_id=f"{doc_id}:rationale:{rationale_index}",
                    text=text,
                    metadata=FrozenAttributes.from_mapping({
                        "doc_id": doc_id,
                        "title": document.get("title"),
                        "sentence_ids": sentence_ids,
                    }),
                ))
        if len(labels) > 1:
            raise ValueError(f"SciFact claim {raw_id} has conflicting SUPPORT/CONTRADICT evidence")
        raw_label = next(iter(labels), "NOT_ENOUGH_INFO")
        support = SCIFACT_SUPPORT_MAP[raw_label]

        cited_doc_ids = tuple(str(value) for value in (record.get("cited_doc_ids") or []))
        missing_cited_docs = tuple(doc_id for doc_id in cited_doc_ids if doc_id not in corpus)
        if missing_cited_docs:
            raise ValueError(f"SciFact claim {raw_id} cites missing documents: {missing_cited_docs}")
        claim_id = f"scifact:{raw_id}"
        return BenchmarkClaim(
            claim_id=claim_id,
            text=claim_text,
            provenance=Provenance(
                dataset=self.dataset_name,
                source_id=str(raw_id),
                citation_ids=cited_doc_ids,
                metadata=FrozenAttributes.from_mapping({
                    "split": self.split,
                    "evidence_document_ids": list(native_evidence),
                    "source_group_ids": list(cited_doc_ids),
                }),
            ),
            assessment=GoldAssessment(
                unit_id=claim_id,
                support=support,
                raw_label=raw_label,
                assessor="SciFact native annotation",
                metadata=FrozenAttributes.from_mapping({
                    "label_mapping": "SUPPORT=>FULL; CONTRADICT/NEI=>NO",
                }),
            ),
            evidence=EvidenceBundle(tuple(items)) if items else None,
            metadata=FrozenAttributes.from_mapping({
                "word_count": len(claim_text.split()),
                "cited_document_count": len(cited_doc_ids),
                "gold_rationale_count": len(items),
                "native_evidence_text_available": bool(items),
            }),
        )

    def load_claims(self, path: str | Path, *, limit: int | None = None) -> list[BenchmarkClaim]:
        output: list[BenchmarkClaim] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    output.append(self.from_record(json.loads(line)))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid SciFact claim at line {line_number}: {exc}") from exc
                if limit is not None and len(output) >= limit:
                    break
        return output
