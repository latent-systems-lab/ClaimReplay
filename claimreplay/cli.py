"""Minimal CLI for the first end-to-end vertical slice."""

from __future__ import annotations

import argparse
import json

from .io import load_response
from .policies import FeatureScorePolicy, PrivilegedOraclePolicy, RandomPolicy
from .replay import replay
from .tasks import review_need_detection, unsupported_detection


def _budget(value: str) -> int | float:
    return float(value) if any(marker in value.lower() for marker in (".", "e")) else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response")
    parser.add_argument("--policy", choices=("random", "feature", "oracle"), default="random")
    parser.add_argument("--feature", default="priority")
    parser.add_argument("--task", choices=("unsupported", "review-needed"), default="unsupported")
    parser.add_argument("--budget", type=_budget, required=True, help="Integer count or decimal fraction in (0,1]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    response = load_response(args.response)
    policy = {
        "random": RandomPolicy(),
        "feature": FeatureScorePolicy(args.feature),
        "oracle": PrivilegedOraclePolicy(),
    }[args.policy]
    task = unsupported_detection() if args.task == "unsupported" else review_need_detection()
    trace = replay(response, policy, task, args.budget, seed=args.seed)
    print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
