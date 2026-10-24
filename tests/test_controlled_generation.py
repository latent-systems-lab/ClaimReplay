from __future__ import annotations

from dataclasses import replace

import pytest

from claimreplay.controlled import ControlledResponseBuilder, ControlledResponseConfig
from claimreplay.generation import (
    ExactTextSemanticValidator,
    GenerationCache,
    GenerationRunner,
    GenerationProvider,
    MockGenerationProvider,
    ProviderResponse,
    ReviewDecisionValue,
    RetryPolicy,
    SemanticReviewDecision,
    SemanticStatus,
    adjudicate_generation_artifact,
    export_semantic_review_queue,
    freeze_controlled_response,
    request_from_manifest,
)
from claimreplay.models import (
    BenchmarkClaim,
    FrozenAttributes,
    GoldAssessment,
    Provenance,
    SupportLevel,
)
from claimreplay.policies import FeatureScorePolicy
from claimreplay.replay import replay
from claimreplay.tasks import unsupported_detection
from claimreplay.validation import assert_policy_view_gold_free


def make_claims(count: int = 180) -> list[BenchmarkClaim]:
    claims = []
    for index in range(count):
        support = SupportLevel.NO_SUPPORT if index % 3 == 0 else SupportLevel.FULL_SUPPORT
        claim_id = f"synthetic:{index:04d}"
        claims.append(BenchmarkClaim(
            claim_id=claim_id,
            text=f"Synthetic proposition number {index}.",
            provenance=Provenance(
                dataset="synthetic_benchmark",
                source_id=str(index),
                metadata=FrozenAttributes.from_mapping({"source_group_ids": [f"source:{index:04d}"]}),
            ),
            assessment=GoldAssessment(
                unit_id=claim_id,
                support=support,
                raw_label="NEGATIVE" if support is SupportLevel.NO_SUPPORT else "POSITIVE",
            ),
        ))
    return claims


@pytest.mark.parametrize(
    ("size", "prevalence", "expected"),
    [(20, "0.05", 1), (20, "0.10", 2), (20, "0.20", 4),
     (40, "0.05", 2), (40, "0.10", 4), (40, "0.20", 8)],
)
def test_exact_controlled_prevalence_and_deterministic_sampling(size, prevalence, expected):
    builder = ControlledResponseBuilder(make_claims(), unsupported_detection())
    config = ControlledResponseConfig("group-a", size, prevalence, seed=47, split="test")
    first = builder.build(config)
    second = builder.build(config)
    assert first.to_dict() == second.to_dict()
    assert first.manifest_hash == second.manifest_hash
    assert len(first.entries) == size
    assert sum(entry.claim.assessment.support is SupportLevel.NO_SUPPORT for entry in first.entries) == expected


def test_build_many_prevents_claim_and_source_overlap():
    builder = ControlledResponseBuilder(make_claims(), unsupported_detection())
    manifests = builder.build_many([
        ControlledResponseConfig("cell-a", 20, "0.05", seed=1, split="test"),
        ControlledResponseConfig("cell-b", 20, "0.10", seed=2, split="test"),
        ControlledResponseConfig("cell-c", 20, "0.20", seed=3, split="test"),
    ])
    claim_sets = [{entry.claim.claim_id for entry in manifest.entries} for manifest in manifests]
    source_sets = [{group for entry in manifest.entries for group in entry.source_group_ids} for manifest in manifests]
    for left in range(len(manifests)):
        for right in range(left + 1, len(manifests)):
            assert claim_sets[left].isdisjoint(claim_sets[right])
            assert source_sets[left].isdisjoint(source_sets[right])


def test_infeasible_controlled_configuration_has_clear_error():
    builder = ControlledResponseBuilder(make_claims(10), unsupported_detection())
    with pytest.raises(ValueError, match="Infeasible controlled configuration"):
        builder.build(ControlledResponseConfig("too-large", 20, "0.20", seed=1, split="test"))


def test_mock_generation_cache_and_freeze_are_offline_and_reproducible(tmp_path):
    manifest = ControlledResponseBuilder(make_claims(), unsupported_detection()).build(
        ControlledResponseConfig("cell-a", 20, "0.10", seed=9, split="test")
    )
    request = request_from_manifest(manifest, provider="mock", model="exact-copy")
    serialized = str(request.to_dict()).upper()
    assert "NO_SUPPORT" not in serialized
    assert "FULL_SUPPORT" not in serialized
    assert "RAW_LABEL" not in serialized
    assert "SOURCE_CLAIM_ID" not in serialized
    assert manifest.entries[0].claim.claim_id.upper() not in serialized

    provider = MockGenerationProvider(mode="exact")
    runner = GenerationRunner(
        provider,
        ExactTextSemanticValidator(),
        cache=GenerationCache(tmp_path / "cache"),
    )
    first = runner.run(request)
    second = runner.run(request)
    assert first.status is SemanticStatus.ACCEPTED
    assert second.status is SemanticStatus.ACCEPTED
    assert second.from_cache is True
    assert provider.call_count == 1

    response = freeze_controlled_response(manifest, first)
    assert response.metadata.get("not_naturalistic_rag_answer") is True
    assert all(unit.evidence is None for unit in response.units)
    assert_policy_view_gold_free(response.policy_view())
    assert response.policy_view() == response.policy_view()


def test_changed_realization_requires_review_and_cannot_be_frozen():
    manifest = ControlledResponseBuilder(make_claims(), unsupported_detection()).build(
        ControlledResponseConfig("cell-a", 20, "0.10", seed=9, split="test")
    )
    request = request_from_manifest(manifest, provider="mock", model="prefix")
    artifact = GenerationRunner(
        MockGenerationProvider(mode="prefix"), ExactTextSemanticValidator()
    ).run(request)
    assert artifact.status is SemanticStatus.NEEDS_REVIEW
    assert artifact.exclusion_reason == "semantic faithfulness requires explicit review"
    assert all(row.source_text and row.generated_text for row in artifact.validations)
    with pytest.raises(ValueError, match="Only ACCEPTED"):
        freeze_controlled_response(manifest, artifact)


def test_semantic_review_queue_requires_explicit_complete_adjudication(tmp_path):
    manifest = ControlledResponseBuilder(make_claims(), unsupported_detection()).build(
        ControlledResponseConfig("cell-a", 20, "0.10", seed=9, split="test")
    )
    request = request_from_manifest(manifest, provider="mock", model="prefix")
    artifact = GenerationRunner(
        MockGenerationProvider(mode="prefix"), ExactTextSemanticValidator()
    ).run(request)
    queue = tmp_path / "semantic_review.jsonl"
    assert export_semantic_review_queue(artifact, queue) == 20
    assert "NO_SUPPORT" not in queue.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        adjudicate_generation_artifact(artifact, [])
    accepted = adjudicate_generation_artifact(artifact, [
        SemanticReviewDecision(row.item_id, ReviewDecisionValue.ACCEPT, "reviewer-a", "faithful")
        for row in artifact.validations
    ])
    assert accepted.status is SemanticStatus.ACCEPTED
    assert all(row.validator == "human-review:reviewer-a" for row in accepted.validations)
    assert len(freeze_controlled_response(manifest, accepted).units) == 20


def test_invalid_empty_realization_is_rejected():
    manifest = ControlledResponseBuilder(make_claims(), unsupported_detection()).build(
        ControlledResponseConfig("cell-a", 20, "0.10", seed=9, split="test")
    )
    request = request_from_manifest(manifest, provider="mock", model="empty")
    artifact = GenerationRunner(
        MockGenerationProvider(mode="empty"), ExactTextSemanticValidator()
    ).run(request)
    assert artifact.status is SemanticStatus.REJECTED
    assert "empty" in artifact.exclusion_reason


def test_gold_mutation_cannot_change_policy_view_ranking_or_generation_cache_key():
    manifest = ControlledResponseBuilder(make_claims(), unsupported_detection()).build(
        ControlledResponseConfig("cell-a", 20, "0.10", seed=9, split="test")
    )
    request = request_from_manifest(manifest, provider="mock", model="exact")
    artifact = GenerationRunner(
        MockGenerationProvider(mode="exact"), ExactTextSemanticValidator()
    ).run(request)
    response = freeze_controlled_response(manifest, artifact)
    changed_assessments = tuple(
        replace(
            row,
            support=(SupportLevel.NO_SUPPORT if row.support is SupportLevel.FULL_SUPPORT else SupportLevel.FULL_SUPPORT),
            raw_label="MUTATED",
        )
        for row in response.gold_assessments
    )
    changed = replace(response, gold_assessments=changed_assessments)
    policy = FeatureScorePolicy("word_count")
    assert response.policy_view() == changed.policy_view()
    assert policy.rank(response.policy_view(), seed=3) == policy.rank(changed.policy_view(), seed=3)
    assert response.response_hash == changed.response_hash
    assert response.gold_hash != changed.gold_hash
    mutated_entries = tuple(
        replace(
            entry,
            claim=replace(
                entry.claim,
                assessment=replace(
                    entry.claim.assessment,
                    support=(SupportLevel.NO_SUPPORT if entry.claim.assessment.support is SupportLevel.FULL_SUPPORT else SupportLevel.FULL_SUPPORT),
                    raw_label="MUTATED",
                ),
            ),
        )
        for entry in manifest.entries
    )
    mutated_manifest = replace(manifest, entries=mutated_entries)
    assert request.cache_key == request_from_manifest(mutated_manifest, provider="mock", model="exact").cache_key
    assert replay(response, policy, unsupported_detection(), 2).selected_unit_ids == replay(
        changed, policy, unsupported_detection(), 2
    ).selected_unit_ids


def test_explicit_retry_policy_retries_schema_failure_then_accepts():
    class FlakyProvider(GenerationProvider):
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def generate(self, request, *, attempt):
            self.calls += 1
            exact = MockGenerationProvider(mode="exact").generate(request, attempt=attempt)
            if attempt == 1:
                return replace(exact, provider=self.name, units=exact.units[:-1])
            return replace(exact, provider=self.name)

    manifest = ControlledResponseBuilder(make_claims(), unsupported_detection()).build(
        ControlledResponseConfig("cell-a", 20, "0.10", seed=9, split="test")
    )
    request = request_from_manifest(manifest, provider="flaky", model="exact")
    provider = FlakyProvider()
    artifact = GenerationRunner(
        provider,
        ExactTextSemanticValidator(),
        retry_policy=RetryPolicy(max_attempts=2),
    ).run(request)
    assert provider.calls == 2
    assert artifact.attempts == 2
    assert artifact.status is SemanticStatus.ACCEPTED
