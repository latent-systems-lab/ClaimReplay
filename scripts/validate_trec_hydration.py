"""Validate user-provided TREC RAG 2024 and evidence assets without download."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from claimreplay.adapters import TrecRag2024Adapter
from claimreplay.evidence import EvidenceTextCache, materialize_response_evidence


EXPECTED = {
    "without_prediction": {"sha256": "96a55934dfbfd5c2350d05f83faffe236976e9404931725f014b9107f4f5e373", "topics": 9, "responses": 381, "units": 3406},
    "with_prediction": {"sha256": "3224b7f2fadf3635b54f9f87320d557832234253a76600724511a7b1c3724369", "topics": 14, "responses": 585, "units": 5020},
    "webassess": {"sha256": "76fd9641fcd6699bdfc601a834c0e3c85f3a8a6a2b49b7c484eadf12bab280ba", "topics": 13, "responses": 549, "units": 4737},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trec-dir", type=Path, required=True)
    parser.add_argument("--primary-evidence-cache", type=Path, required=True)
    parser.add_argument("--supplementary-evidence-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {}
    checks = []
    for condition, expected in EXPECTED.items():
        source = args.trec_dir / f"{condition}.jsonl"
        responses = TrecRag2024Adapter(judgment_condition=condition).load(source)
        cache_path = args.primary_evidence_cache if condition == "without_prediction" else args.supplementary_evidence_cache
        cache = EvidenceTextCache.load_jsonl(cache_path)
        missing = set()
        for response in responses:
            _, report = materialize_response_evidence(response, cache)
            missing.update(report.missing_evidence_ids)
        observed = {
            "source": source.as_posix(), "source_bytes": source.stat().st_size, "source_sha256": sha(source),
            "topics": len({str(response.metadata.get('topic_id')) for response in responses}),
            "responses": len(responses), "units": sum(len(response.units) for response in responses),
            "evidence_cache": cache_path.as_posix(), "evidence_cache_sha256": sha(cache_path),
            "evidence_cache_entries": len(cache), "missing_evidence_ids": len(missing),
        }
        passed = (observed["source_sha256"] == expected["sha256"] and
                  all(observed[key] == expected[key] for key in ("topics", "responses", "units")) and not missing)
        conditions[condition] = {**observed, "passed": passed}
        checks.append(passed)
    payload = {
        "schema_version": "claimreplay.trec-hydration-validation.v1",
        "status": "PASS" if all(checks) else "FAIL",
        "automatic_download_performed": False,
        "restricted_assets_redistributed": False,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
