from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from claimreplay.evidence import EvidenceTextCache, materialize_response_evidence
from claimreplay.evidence_policies import (
    DeterministicMockNliScorer,
    EvidenceScoreCache,
    NliEvidenceRiskPolicy,
)
from claimreplay.models import (
    EvidenceBundle,
    EvidenceItem,
    FrozenAttributes,
    FrozenResponse,
    GoldAssessment,
    Provenance,
    SupportLevel,
    VerificationUnit,
)
from claimreplay.replay import replay
from claimreplay.tasks import unsupported_detection
from claimreplay.validation import assert_policy_view_gold_free


def response_with_evidence() -> FrozenResponse:
    units = (
        VerificationUnit("u1", "Cats sleep.", Provenance("toy"), evidence=EvidenceBundle((EvidenceItem("e1", "Cats sleep. Animals rest."),))),
        VerificationUnit("u2", "Mars is green.", Provenance("toy"), evidence=EvidenceBundle((EvidenceItem("e2", "Mars is known as the red planet."),))),
        VerificationUnit("u3", "No citation.", Provenance("toy"), evidence=None),
    )
    return FrozenResponse(
        "evidence-toy", units,
        (
            GoldAssessment("u1", SupportLevel.FULL_SUPPORT),
            GoldAssessment("u2", SupportLevel.NO_SUPPORT),
            GoldAssessment("u3", SupportLevel.UNASSESSED),
        ),
        FrozenAttributes.from_mapping({"topic_id": "t1", "run_id": "r1", "policy_visible": {}}),
    )


def test_nli_evidence_policy_ranking_cost_cache_and_gold_isolation(tmp_path):
    response = response_with_evidence()
    cache_path = tmp_path / "scores.json"
    policy = NliEvidenceRiskPolicy(DeterministicMockNliScorer(), cache=EvidenceScoreCache(cache_path))
    first = replay(response, policy, unsupported_detection(), 1)
    assert [row.unit_id for row in first.ranking] == ["u3", "u2", "u1"]
    assert first.triage_cost.score_evaluations == 2
    assert first.triage_cost.model_calls == 1
    assert first.triage_cost.cache_hits == 0
    assert first.triage_cost.cache_misses == 2

    changed_gold = replace(response, gold_assessments=tuple(
        replace(row, support=SupportLevel.FULL_SUPPORT) for row in response.gold_assessments
    ))
    second = replay(changed_gold, policy, unsupported_detection(), 1)
    assert second.ranking == first.ranking
    assert second.response_hash == first.response_hash
    assert second.gold_hash != first.gold_hash
    assert second.triage_cost.model_calls == 0
    assert second.triage_cost.cache_hits == 2
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache_payload["contains_gold"] is False
    cache_entries = json.dumps(cache_payload["entries"]).lower()
    assert "gold" not in cache_entries
    assert "support" not in cache_entries


def test_evidence_policy_view_is_gold_free():
    view = response_with_evidence().evidence_policy_view()
    assert_policy_view_gold_free(view)
    assert not hasattr(view, "gold_assessments")
    assert not hasattr(view, "gold_hash")
    assert all(not hasattr(unit, "provenance") for unit in view.units)
    assert view.units[0].evidence_texts == ("Cats sleep. Animals rest.",)


def test_evidence_cache_materializes_text_and_verifies_hash(tmp_path):
    text = "Materialized official segment."
    row = {
        "evidence_id": "e-missing",
        "text": text,
        "source": {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "tar_member": "fixture.gz"},
    }
    path = tmp_path / "evidence.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cache = EvidenceTextCache.load_jsonl(path)
    response = FrozenResponse(
        "materialize", (VerificationUnit(
            "u1", "Claim.", Provenance("toy"), evidence=EvidenceBundle((EvidenceItem("e-missing"), EvidenceItem("not-found")))
        ),), (GoldAssessment("u1", SupportLevel.UNASSESSED),),
        FrozenAttributes.from_mapping({"policy_visible": {}}),
    )
    updated, report = materialize_response_evidence(response, cache)
    assert updated.units[0].evidence.items[0].text == text
    assert updated.units[0].evidence.items[1].text is None
    assert report.materialized_item_count == 1
    assert report.missing_evidence_ids == ("not-found",)
