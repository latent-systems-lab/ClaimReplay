"""Dependency-free release verification; run after installing the wheel."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from claimreplay.io import load_response
from claimreplay.models import FrozenAttributes
from claimreplay.policies import FeatureScorePolicy
from claimreplay.replay import replay
from claimreplay.tasks import unsupported_detection


ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "fixtures" / "toy_response.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: list[str] = []
    require(importlib.metadata.version("claimreplay") == "0.6.0rc2", "installed version mismatch")
    checks.append("installed-version")
    response = load_response(TOY)
    trace = replay(response, FeatureScorePolicy("priority"), unsupported_detection(), 2)
    require(trace.selected_unit_ids == ("u03", "u10") and trace.metrics.recall == 1.0, "toy metrics mismatch")
    checks.append("toy-metrics")

    injected = replace(response, units=(replace(response.units[0],
        observable_features=FrozenAttributes.from_mapping({"gold_label": 1})),) + response.units[1:])
    try:
        replay(injected, FeatureScorePolicy("gold_label"), unsupported_detection(), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("gold-like feature was not rejected")
    checks.append("gold-field-rejection")

    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = ROOT / row["path"]
        require(path.is_file(), f"missing manifest member {row['path']}")
        require(path.stat().st_size == row["bytes"], f"size mismatch {row['path']}")
        require(sha256(path) == row["sha256"], f"hash mismatch {row['path']}")
    checks.append("release-manifest")

    with tempfile.TemporaryDirectory(prefix="claimreplay-release-test-") as temporary:
        first, second = Path(temporary) / "first", Path(temporary) / "second"
        script = ROOT / "examples" / "run_external_policy_tutorial.py"
        subprocess.run([sys.executable, str(script), str(TOY), str(first)], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, str(script), str(TOY), str(second), "--compare-to", str(first)
        ], cwd=ROOT, check=True)
        receipt = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
        require(receipt["trace_count"] == 9, "tutorial trace count mismatch")
        require(all(item["match"] for item in receipt["comparison"].values()), "tutorial mismatch")
    checks.append("external-policy-determinism")

    names = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    require("examples/fixtures/trec_rag_2024_sample.jsonl" not in names, "restricted fixture bundled")
    require(not any(name.startswith("public_results/") and ("traces" in name or "scores.jsonl" in name)
                    for name in names), "unit-level result bundled")
    checks.append("information-boundary-inventory")
    print(json.dumps({"passed": True, "checks": checks, "check_count": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
