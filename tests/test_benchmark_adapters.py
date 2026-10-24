from __future__ import annotations

import json

import pytest

from claimreplay.adapters import FeverousAdapter, SciFactAdapter
from claimreplay.models import BenchmarkClaim, SupportLevel


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_feverous_retains_native_labels_and_id_only_evidence(tmp_path):
    path = tmp_path / "feverous.jsonl"
    write_jsonl(path, [
        {"id": "", "claim": "", "label": "", "evidence": ""},
        {
            "id": 7,
            "claim": "A supported claim.",
            "label": "SUPPORTS",
            "evidence": [{
                "content": ["Page_sentence_1", "Page_sentence_1"],
                "context": {"Page_sentence_1": ["Page_title"]},
            }],
            "challenge": "Other",
        },
        {
            "id": 8,
            "claim": "An insufficiently supported claim.",
            "label": "NOT ENOUGH INFO",
            "evidence": [],
        },
    ])
    claims = FeverousAdapter(split="test").load_claims(path)
    assert len(claims) == 2
    assert all(isinstance(claim, BenchmarkClaim) for claim in claims)
    assert claims[0].assessment.support is SupportLevel.FULL_SUPPORT
    assert claims[1].assessment.support is SupportLevel.NO_SUPPORT
    assert claims[1].assessment.raw_label == "NOT ENOUGH INFO"
    assert claims[0].evidence is not None
    assert len(claims[0].evidence.items) == 1
    assert claims[0].evidence.items[0].text is None
    assert claims[0].evidence.items[0].metadata.get("occurrences") == [
        {"evidence_set_index": 0, "context_ids": ["Page_title"]},
        {"evidence_set_index": 0, "context_ids": ["Page_title"]},
    ]
    assert not hasattr(claims[0], "policy_view")


def test_scifact_joins_native_rationale_sentences_and_preserves_nei(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    write_jsonl(corpus_path, [{
        "doc_id": 10,
        "title": "Evidence paper",
        "abstract": ["Sentence zero.", "Sentence one.", "Sentence two."],
        "structured": False,
    }])
    write_jsonl(claims_path, [
        {
            "id": 1,
            "claim": "Supported.",
            "evidence": {"10": [{"label": "SUPPORT", "sentences": [0, 2]}]},
            "cited_doc_ids": [10],
        },
        {"id": 2, "claim": "Unknown.", "evidence": {}, "cited_doc_ids": [10]},
    ])
    claims = SciFactAdapter(corpus_path, split="dev").load_claims(claims_path)
    assert claims[0].assessment.support is SupportLevel.FULL_SUPPORT
    assert claims[0].evidence.items[0].text == "Sentence zero. Sentence two."
    assert claims[0].evidence.items[0].metadata.get("sentence_ids") == [0, 2]
    assert claims[1].assessment.support is SupportLevel.NO_SUPPORT
    assert claims[1].assessment.raw_label == "NOT_ENOUGH_INFO"
    assert claims[1].evidence is None
    assert claims[1].provenance.citation_ids == ("10",)


def test_scifact_rejects_missing_documents_and_bad_sentence_indices(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    write_jsonl(corpus_path, [{
        "doc_id": 10,
        "title": "Evidence paper",
        "abstract": ["Only sentence."],
        "structured": False,
    }])
    write_jsonl(claims_path, [{
        "id": 1,
        "claim": "Broken rationale.",
        "evidence": {"10": [{"label": "CONTRADICT", "sentences": [3]}]},
        "cited_doc_ids": [10],
    }])
    with pytest.raises(ValueError, match="out-of-range sentence"):
        SciFactAdapter(corpus_path).load_claims(claims_path)


def test_scifact_rejects_claim_level_label_conflicts(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    write_jsonl(corpus_path, [
        {"doc_id": 10, "title": "A", "abstract": ["A."], "structured": False},
        {"doc_id": 11, "title": "B", "abstract": ["B."], "structured": False},
    ])
    write_jsonl(claims_path, [{
        "id": 1,
        "claim": "Conflicted.",
        "evidence": {
            "10": [{"label": "SUPPORT", "sentences": [0]}],
            "11": [{"label": "CONTRADICT", "sentences": [0]}],
        },
        "cited_doc_ids": [10, 11],
    }])
    with pytest.raises(ValueError, match="conflicting"):
        SciFactAdapter(corpus_path).load_claims(claims_path)
