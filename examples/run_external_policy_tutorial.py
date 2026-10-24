"""Run the public ClaimLength tutorial and export deterministic replay artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from claim_length_policy import ClaimLength
from claimreplay.io import load_response
from claimreplay.policies import LexicalCheckworthinessPolicy, RandomPolicy
from claimreplay.replay import replay
from claimreplay.tasks import review_need_detection


BUDGETS = (0.05, 0.10, 0.20)
SEED = 1729
DETERMINISTIC_FILES = ("metrics.csv", "rankings.csv", "selected_units.csv", "replay_traces.jsonl")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare-to", type=Path, help="Prior output directory to compare")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    response = load_response(args.fixture)
    task = review_need_detection()
    policies = (ClaimLength(), RandomPolicy(), LexicalCheckworthinessPolicy())
    traces: list[dict] = []
    metrics: list[dict] = []
    rankings: list[dict] = []
    selections: list[dict] = []
    text_by_id = {unit.unit_id: unit.text for unit in response.units}

    for policy in policies:
        for requested_budget in BUDGETS:
            trace = replay(response, policy, task, requested_budget, seed=SEED)
            payload = trace.to_dict()
            traces.append(payload)
            common = {
                "response_id": trace.response_id,
                "policy": trace.policy_name,
                "requested_budget": requested_budget,
                "resolved_budget": trace.resolved_budget,
                "realized_fraction": trace.resolved_budget / len(response.units),
                "seed": trace.seed,
            }
            metrics.append(common | trace.metrics.as_dict())
            rankings.extend(common | {
                "rank": rank,
                "unit_id": item.unit_id,
                "score": item.score,
            } for rank, item in enumerate(trace.ranking, 1))
            selections.extend(common | {
                "selection_rank": rank,
                "unit_id": unit_id,
                "text": text_by_id[unit_id],
            } for rank, unit_id in enumerate(trace.selected_unit_ids, 1))

    write_csv(args.output / "metrics.csv", metrics)
    write_csv(args.output / "rankings.csv", rankings)
    write_csv(args.output / "selected_units.csv", selections)
    (args.output / "replay_traces.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in traces), encoding="utf-8"
    )
    file_hashes = {name: sha256(args.output / name) for name in DETERMINISTIC_FILES}
    receipt = {
        "schema": "claimreplay.external-policy-tutorial.v1",
        "fixture_sha256": sha256(args.fixture),
        "response_hash": response.response_hash,
        "trace_count": len(traces),
        "budgets": list(BUDGETS),
        "seed": SEED,
        "deterministic_files": file_hashes,
        "paid_api_calls": 0,
        "neural_model_calls": 0,
    }
    if args.compare_to:
        receipt["comparison"] = {
            name: {
                "expected": sha256(args.compare_to / name),
                "actual": digest,
                "match": sha256(args.compare_to / name) == digest,
            }
            for name, digest in file_hashes.items()
        }
        if not all(row["match"] for row in receipt["comparison"].values()):
            raise SystemExit("Deterministic output mismatch")
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "trace_count": len(traces),
        "selected_assessment_coverage": sorted({row["SelectedAssessmentCoverage"] for row in metrics}),
        "deterministic_comparison": None if not args.compare_to else "passed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
