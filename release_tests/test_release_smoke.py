from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest

from claimreplay.evidence_policies import DeterministicMockNliScorer, EvidenceScoreCache, NliEvidenceRiskPolicy
from claimreplay.io import load_response
from claimreplay.models import EvidenceBundle, EvidenceItem, FrozenAttributes, SupportLevel
from claimreplay.policies import FeatureScorePolicy, PrivilegedOraclePolicy, RandomPolicy
from claimreplay.replay import replay
from claimreplay.tasks import review_need_detection, unsupported_detection


ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "fixtures" / "toy_response.json"


def test_installed_release_identity_and_toy_metrics():
    assert importlib.metadata.version("claimreplay") == "0.6.0rc2"
    response = load_response(TOY)
    trace = replay(response, FeatureScorePolicy("priority"), unsupported_detection(), 2)
    assert trace.selected_unit_ids == ("u03", "u10")
    assert trace.metrics.found == 2
    assert trace.metrics.recall == 1.0
    assert trace.metrics.selected_assessment_coverage == 1.0


def test_gold_mutation_does_not_change_ordinary_ranking():
    response = load_response(TOY)
    changed = replace(response, gold_assessments=tuple(
        replace(row, support=SupportLevel.NO_SUPPORT if row.support is not SupportLevel.NO_SUPPORT else SupportLevel.FULL_SUPPORT)
        for row in response.gold_assessments
    ))
    policy = RandomPolicy()
    assert policy.rank(response.policy_view(), seed=1729) == policy.rank(changed.policy_view(), seed=1729)


def test_direct_gold_feature_rejected_but_legitimate_feature_and_oracle_work():
    response = load_response(TOY)
    injected = replace(response, units=(replace(response.units[0],
        observable_features=FrozenAttributes.from_mapping({"gold_label": 1})),) + response.units[1:])
    with pytest.raises(ValueError, match="may reveal"):
        replay(injected, FeatureScorePolicy("gold_label"), unsupported_detection(), 1)
    assert replay(response, FeatureScorePolicy("priority"), unsupported_detection(), 1).resolved_budget == 1
    assert replay(response, PrivilegedOraclePolicy(), unsupported_detection(), 1).policy_name == "oracle"


def test_first_evidence_position_and_cache_v2(tmp_path):
    response = load_response(TOY)
    unit = replace(response.units[0], evidence=EvidenceBundle((
        EvidenceItem("first", None), EvidenceItem("second", "later evidence"),
    )))
    response = replace(response, units=(unit,) + response.units[1:])
    path = tmp_path / "cache.json"
    trace = replay(response, NliEvidenceRiskPolicy(
        DeterministicMockNliScorer(), cache=EvidenceScoreCache(path)
    ), review_need_detection(), 1)
    assert next(row for row in trace.ranking if row.unit_id == unit.unit_id).score == 1.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "claimreplay.evidence-score-cache.v2"
    assert payload["contains_gold"] is False


def test_external_tutorial_is_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    script = ROOT / "examples" / "run_external_policy_tutorial.py"
    subprocess.run([sys.executable, str(script), str(TOY), str(first)], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(script), str(TOY), str(second), "--compare-to", str(first)
    ], cwd=ROOT, check=True)
    receipt = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["trace_count"] == 9
    assert all(row["match"] for row in receipt["comparison"].values())


def test_release_tree_excludes_restricted_and_unit_oracle_assets():
    # Inspect the immutable package inventory, not the live extracted tree.
    # Python/pytest legitimately creates __pycache__ while this suite runs;
    # those runtime files are not members of the release archive.
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    names = {row["path"] for row in manifest["files"]}
    assert "examples/fixtures/trec_rag_2024_sample.jsonl" not in names
    forbidden_parts = {"tmp", "venv", ".venv", "__pycache__", ".claimreplay-git"}
    assert not any(forbidden_parts & set(Path(name).parts) for name in names)
    assert not any("replay_traces" in name and name.startswith("public_results/") for name in names)
    assert not any("scores.jsonl" in name and name.startswith("public_results/") for name in names)


def test_release_manifest_matches_when_present():
    path = ROOT / "RELEASE_MANIFEST.json"
    if not path.exists():
        pytest.skip("manifest is generated in the staged release candidate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["files"]:
        member = ROOT / row["path"]
        assert member.is_file()
        assert member.stat().st_size == row["bytes"]
        assert hashlib.sha256(member.read_bytes()).hexdigest() == row["sha256"]
