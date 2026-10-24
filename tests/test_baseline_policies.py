from __future__ import annotations

from claimreplay.adapters import TrecRag2024Adapter
from claimreplay.models import FrozenAttributes, PolicyView, PolicyVisibleUnit
from claimreplay.policies import CitationScarcityPolicy, LexicalCheckworthinessPolicy


def test_citation_scarcity_uses_only_visible_feature_and_reports_cost():
    response = TrecRag2024Adapter().load_jsonl(
        "examples/fixtures/trec_rag_2024_synthetic.jsonl", limit=1
    )[0]
    view = response.policy_view()
    policy = CitationScarcityPolicy()
    ranking = policy.rank(view, seed=47)
    assert ranking[0].unit_id == "s0003"
    assert ranking[0].score == 0.0
    assert policy.triage_cost(view).to_dict() == {
        "units_considered": 4,
        "score_evaluations": 4,
        "model_calls": 0,
        "retrieval_calls": 0,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
    }


def test_lexical_checkworthiness_is_deterministic_and_model_free():
    view = PolicyView(
        response_id="policy-view:test",
        units=(
            PolicyVisibleUnit("short", "A brief statement."),
            PolicyVisibleUnit("specific", "Report 2024 names Warsaw and Poland in 17 findings."),
        ),
        metadata=FrozenAttributes(),
        response_hash="test",
    )
    policy = LexicalCheckworthinessPolicy()
    first = policy.rank(view, seed=1)
    second = policy.rank(view, seed=999)
    assert first == second
    assert first[0].unit_id == "specific"
    cost = policy.triage_cost(view)
    assert cost.model_calls == 0
    assert cost.retrieval_calls == 0
    assert cost.score_evaluations == 2
    assert cost.input_tokens == 12
