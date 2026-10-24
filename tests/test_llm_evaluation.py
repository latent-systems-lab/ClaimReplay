from __future__ import annotations

import json
from dataclasses import replace

import pytest

from claimreplay.llm import (
    ExecutionLimits,
    LlmBudgetExceeded,
    LlmInferenceError,
    MockStructuredProvider,
    ProviderConfiguration,
    StructuredResponseCache,
    content_hash,
    execute_structured_requests,
)
from claimreplay.llm_evaluation import (
    RecordedLlmPriorityPolicy,
    RecordedPriorityScore,
    judge_request,
    priority_request,
    validate_judge_output,
    validate_priority_output,
)
from claimreplay.models import FrozenAttributes, FrozenResponse, GoldAssessment, Provenance, SupportLevel, VerificationUnit
from claimreplay.replay import replay
from claimreplay.tasks import review_need_detection


PRIORITY_PROMPT = "Assign a review-priority score. Return only the required JSON object."
JUDGE_PROMPT = "Classify support using the supplied claim and evidence. Return JSON."


def _response(labels=(SupportLevel.FULL_SUPPORT, SupportLevel.NO_SUPPORT)) -> FrozenResponse:
    units = tuple(
        VerificationUnit(
            unit_id=f"u{i}",
            text=text,
            provenance=Provenance(dataset="fixture", source_id=f"hidden-{i}"),
        )
        for i, text in enumerate(("A short claim.", "A considerably longer claim with a number 2026."))
    )
    return FrozenResponse(
        response_id="fixture-response",
        units=units,
        gold_assessments=tuple(
            GoldAssessment(unit.unit_id, label, raw_label=f"raw-{label.value}")
            for unit, label in zip(units, labels)
        ),
    )


def test_priority_request_excludes_provenance_gold_and_metadata_from_prompt():
    request = priority_request(
        "native-secret-id",
        "Visible claim text.",
        PRIORITY_PROMPT,
        metadata={"gold_label": "NO_SUPPORT", "oracle_selected": True, "topic_id": "secret"},
    )
    payload = request.provider_payload(ProviderConfiguration(provider="openai", model="snapshot"))
    serialized = json.dumps(payload)
    assert "Visible claim text." in serialized
    assert "native-secret-id" not in serialized
    assert "NO_SUPPORT" not in serialized
    assert "oracle_selected" not in serialized
    assert "topic_id" not in serialized


def test_judge_request_excludes_human_label_and_policy_ranking_metadata():
    first = judge_request("u1", "Claim.", "Evidence.", JUDGE_PROMPT,
                          metadata={"human_label": "FULL_SUPPORT", "policy_rank": 1})
    second = judge_request("u1", "Claim.", "Evidence.", JUDGE_PROMPT,
                           metadata={"human_label": "NO_SUPPORT", "policy_rank": 99})
    assert first.input_hash == second.input_hash
    config = ProviderConfiguration(provider="openai", model="x")
    assert first.provider_payload(config) == second.provider_payload(config)


@pytest.mark.parametrize("bad", [{}, {"priority_score": True}, {"priority_score": -1},
                                  {"priority_score": 101}, {"priority_score": 20, "extra": 1}])
def test_priority_schema_rejects_invalid_outputs(bad):
    with pytest.raises(ValueError):
        validate_priority_output(bad)


@pytest.mark.parametrize("label", ["FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT", "ABSTAIN"])
def test_judge_schema_accepts_frozen_labels(label):
    assert validate_judge_output({"support_label": label}) == {"support_label": label}


def test_mock_execution_deduplicates_content_and_reuses_separate_cache(tmp_path):
    requests = [priority_request("u1", "Same text.", PRIORITY_PROMPT),
                priority_request("u2", "Same text.", PRIORITY_PROMPT)]
    provider = MockStructuredProvider(lambda request: {"priority_score": 37})
    cache = StructuredResponseCache(tmp_path / "cache", provider.configuration)
    limits = ExecutionLimits(maximum_requests=10, maximum_planned_tokens=10_000, concurrency=2)
    first = execute_structured_requests(requests, provider, cache, limits, validate_priority_output)
    assert provider.calls == 1
    assert [record.request_id for record in first] == ["u1", "u2"]
    assert all(record.parsed == {"priority_score": 37} for record in first)
    assert len(list((tmp_path / "cache" / "raw").glob("*.json"))) == 1
    assert len(list((tmp_path / "cache" / "parsed").glob("*.json"))) == 1
    second = execute_structured_requests(requests, provider, cache, limits, validate_priority_output)
    assert provider.calls == 1
    assert all(record.cache_hit for record in second)


def test_cache_rejects_truncated_and_configuration_mismatch(tmp_path):
    request = priority_request("u1", "Claim.", PRIORITY_PROMPT)
    provider = MockStructuredProvider(lambda _: {"priority_score": 50})
    cache = StructuredResponseCache(tmp_path / "cache", provider.configuration)
    execute_structured_requests([request], provider, cache, ExecutionLimits(2, 10_000), validate_priority_output)
    next((tmp_path / "cache" / "parsed").glob("*.json")).unlink()
    with pytest.raises(LlmInferenceError, match="Truncated"):
        cache.get(request)
    other = StructuredResponseCache(tmp_path / "other", replace(provider.configuration, model="different"))
    assert other.key(request) != cache.key(request)


def test_request_and_token_ceilings_stop_before_provider_calls(tmp_path):
    provider = MockStructuredProvider(lambda _: {"priority_score": 1})
    cache = StructuredResponseCache(tmp_path / "cache", provider.configuration)
    requests = [priority_request(f"u{i}", f"Different claim {i}.", PRIORITY_PROMPT) for i in range(2)]
    with pytest.raises(LlmBudgetExceeded):
        execute_structured_requests(requests, provider, cache, ExecutionLimits(1, 10_000), validate_priority_output)
    with pytest.raises(LlmBudgetExceeded):
        execute_structured_requests(requests, provider, cache, ExecutionLimits(10, 1), validate_priority_output)
    assert provider.calls == 0


def test_recorded_policy_is_gold_oracle_and_judge_invariant():
    response = _response()
    scores = {
        unit.unit_id: RecordedPriorityScore(unit.unit_id, content_hash(unit.text), score)
        for unit, score in zip(response.units, (10, 90))
    }
    policy = RecordedLlmPriorityPolicy(scores)
    first = replay(response, policy, review_need_detection(), .5, seed=47)
    mutated = replace(
        response,
        gold_assessments=tuple(
            replace(gold, support=SupportLevel.PARTIAL_SUPPORT if gold.support is SupportLevel.FULL_SUPPORT
                    else SupportLevel.FULL_SUPPORT)
            for gold in response.gold_assessments
        ),
        metadata=FrozenAttributes.from_mapping({"oracle_selected": ["u0"], "judge_labels": ["NO_SUPPORT"]}),
    )
    second = replay(mutated, policy, review_need_detection(), .5, seed=47)
    assert first.ranking == second.ranking
    assert first.selected_unit_ids == second.selected_unit_ids == ("u1",)


def test_recorded_policy_rejects_text_identity_change():
    response = _response()
    policy = RecordedLlmPriorityPolicy({
        "u0": RecordedPriorityScore("u0", content_hash("different"), 10),
        "u1": RecordedPriorityScore("u1", content_hash(response.units[1].text), 90),
    })
    with pytest.raises(ValueError, match="input mismatch"):
        replay(response, policy, review_need_detection(), .5)
