import copy
import json
from pathlib import Path

import pytest

from claimreplay.io import load_response
from claimreplay.policies import RandomPolicy
from claimreplay.replay import replay
from claimreplay.reproduce import audit_trace, checked_file, run, sha, summarize
from claimreplay.tasks import review_need_detection


FIXTURE = Path(__file__).resolve().parents[1] / "examples/fixtures/toy_response.json"


def test_checked_input_missing_and_wrong_hash(tmp_path):
    with pytest.raises(ValueError, match="Missing authorized"):
        checked_file(tmp_path, "missing", "0"*64)
    p = tmp_path / "input"; p.write_text("example")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        checked_file(tmp_path, "input", "0"*64)
    assert checked_file(tmp_path, "input", sha(p)) == p


def test_historical_selection_and_metric_tampering():
    response = load_response(FIXTURE); task = review_need_detection()
    row = replay(response, RandomPolicy(), task, .2, seed=47).to_dict()
    assert audit_trace(row, response, task, 47) == row
    bad = copy.deepcopy(row); bad["metrics"]["ReviewNeededFound"] += 1
    with pytest.raises(ValueError, match="metrics differ"):
        audit_trace(bad, response, task, 47)
    bad = copy.deepcopy(row); bad["selected_unit_ids"].reverse()
    with pytest.raises(ValueError, match="selection/quota"):
        audit_trace(bad, response, task, 47)


def test_analytic_reference_is_positive_weighted_within_topic():
    rows = []
    for n, p, b in ((20, 2, 4), (3, 1, 1)):
        rows.append(dict(policy="example", task="review_need_detection", requested_budget=.2,
                         resolved_budget=b, cluster_ids={"topic_id": "same"}, metrics={
                             "ReviewNeededCount": p, "ReviewNeededFound": 1,
                             "SelectedCount": b, "SelectedEligibleCount": b, "TotalUnitCount": n}))
    table, _ = summarize(rows, "topic_id")
    assert table[0]["analytic_random"] == pytest.approx((2*4/20+1/3)/3)
    assert table[0]["recall"] == pytest.approx(2/3)


def test_synthetic_run_repeat_and_no_overwrite(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": "claimreplay.reproduction.v1", "seed": 47,
                                 "budgets": [.05, .1, .2], "synthetic": str(FIXTURE)}))
    a = run(config, tmp_path, tmp_path / "a", "deterministic")
    b = run(config, tmp_path, tmp_path / "b", "deterministic")
    assert a["status"] == b["status"] == "PASS"
    assert a["outputs"] == b["outputs"]
    assert a["api_calls"] == 0
    with pytest.raises(ValueError, match="already exists"):
        run(config, tmp_path, tmp_path / "a", "deterministic")
