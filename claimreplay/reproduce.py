"""No-network reproduction orchestration over existing adapters and replay APIs.

Native inputs and evaluator outputs remain local. This module never calls a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from .adapters import TrecRag2024Adapter
from .cross_dataset import dependency_groups, load_sources, responses as controlled_responses, task as controlled_task
from .io import load_response
from .metrics import compute_metrics
from .policies import RandomPolicy, CitationScarcityPolicy, LexicalCheckworthinessPolicy, PrivilegedOraclePolicy
from .replay import replay, resolve_budget
from .statistics import small_sample_t_interval
from .tasks import review_need_detection, unsupported_detection

VERSION = "claimreplay.reproduction.v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def checked_file(root, value, expected):
    path = root / value
    if not path.is_file():
        raise ValueError(f"Missing authorized input: {value}; see DATA_ACCESS.md")
    if sha(path) != expected:
        raise ValueError(f"Input SHA-256 mismatch: {value}; refusing silent dataset substitution")
    return path


def ingest(config, root):
    """Return separate populations; never pool TREC assessment conditions."""
    populations = []
    for condition, cfg in config.get("trec", {}).items():
        path = checked_file(root, cfg["path"], cfg["sha256"])
        items = TrecRag2024Adapter(judgment_condition=condition).load(path)
        observed = {"responses": len(items), "units": sum(len(r.units) for r in items),
                    "topics": len({str(r.metadata.get('topic_id')) for r in items})}
        if observed != cfg["expected"]:
            raise ValueError(f"{condition}: incompatible counts {observed}")
        tasks = {t.name: t for t in (review_need_detection(), unsupported_detection())}
        chosen = tuple(tasks[name] for name in cfg.get("tasks", tasks))
        if not chosen:
            raise ValueError("Empty evaluation task grid")
        populations.append((condition, items, chosen, "topic_id"))
    controlled = config.get("controlled")
    if controlled:
        for cfg in controlled["datasets"].values():
            checked_file(root, cfg["claims"], cfg["claims_sha256"])
            if "corpus" in cfg:
                checked_file(root, cfg["corpus"], cfg["corpus_sha256"])
        for dataset, source in load_sources(root, controlled).items():
            if source["native_count"] != controlled["datasets"][dataset]["expected_native_count"]:
                raise ValueError(f"Unexpected native record count: {dataset}")
            records = source["records"]
            groups = dependency_groups(records)
            for size in controlled["queue_sizes"]:
                for target in controlled["targets"]:
                    name = f"{dataset}_n{size}_{target}"
                    items = controlled_responses(dataset, records, groups, size, target)
                    populations.append((name, items, (controlled_task(dataset, target),), "group_id"))
    if "synthetic" in config:
        populations.append(("synthetic", [load_response(root / config["synthetic"])],
                            (review_need_detection(), unsupported_detection()), None))
    if not populations:
        raise ValueError("No evaluation populations configured")
    return populations


def audit_trace(row, response, task, seed):
    """Independently recompute metrics from native gold and selected IDs.

    Scores are NOT reconstructed from labels. Source response hashes may include
    evidence materialization absent here; their historical identity is not claimed.
    """
    ids = [r["unit_id"] for r in row["ranking"]]
    expected = {u.unit_id for u in response.units}
    b = resolve_budget(row["requested_budget"], len(expected))
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("Historical ranking membership or uniqueness mismatch")
    if row["resolved_budget"] != b or row["selected_unit_ids"] != ids[:b]:
        raise ValueError("Historical selection/quota mismatch")
    if row["seed"] != seed:
        raise ValueError("Historical seed mismatch")
    recomputed = compute_metrics(response, tuple(ids[:b]), task).as_dict()
    if recomputed != row["metrics"]:
        raise ValueError("Historical metrics differ from native-assessment reconstruction")
    return row


def summarize(traces, cluster_key):
    grouped = defaultdict(list)
    for row in traces:
        grouped[(row["policy"], row["task"], row["requested_budget"])].append(row)
    table, clusters = [], []
    for (policy, task, budget), rows in sorted(grouped.items()):
        prefix = "Unsupported" if task == "unsupported_detection" else "ReviewNeeded"
        sums = defaultdict(lambda: [0, 0, 0.0])
        for row in rows:
            m = row["metrics"]
            key = str(row["cluster_ids"].get(cluster_key, "all")) if cluster_key else "all"
            s = sums[key]
            s[0] += m[prefix + "Found"]
            s[1] += m[prefix + "Count"]
            s[2] += m[prefix + "Count"] * row["resolved_budget"] / m["TotalUnitCount"]
        recalls, randoms, effects = [], [], []
        for cluster, (found, positive, analytic) in sorted(sums.items()):
            recall = found / positive if positive else None
            random = analytic / positive if positive else None
            clusters.append(dict(policy=policy, task=task, budget=budget, cluster=cluster,
                                 found=found, positives=positive, recall=recall, analytic_random=random))
            if positive:
                recalls.append(recall); randoms.append(random); effects.append(recall - random)
        selected = sum(r["metrics"]["SelectedCount"] for r in rows)
        assessed = sum(r["metrics"]["SelectedEligibleCount"] for r in rows)
        total = sum(r["metrics"]["TotalUnitCount"] for r in rows)
        ci = small_sample_t_interval(effects).to_dict() if cluster_key == "topic_id" and 2 <= len(effects) <= 31 else None
        table.append(dict(policy=policy, task=task, budget=budget, responses=len(rows), units=total,
                          estimand="topic_macro" if cluster_key == "topic_id" else "dependency_macro_descriptive" if cluster_key else "single_fixture",
                          estimable_clusters=len(recalls), zero_positive_clusters=len(sums)-len(recalls),
                          recall=fmean(recalls) if recalls else None,
                          analytic_random=fmean(randoms) if randoms else None,
                          difference=fmean(effects) if effects else None,
                          difference_ci_low=ci["confidence_interval"][0] if ci else None,
                          difference_ci_high=ci["confidence_interval"][1] if ci else None,
                          selected=selected, selected_assessed=assessed, selected_unassessed=selected-assessed,
                          selected_coverage=assessed/selected if selected else None, realized_fraction=selected/total))
    return table, clusters


def csv_write(path, rows):
    if not rows:
        raise ValueError("Empty result table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def figure(path, table, population):
    """Dependency-free, editable SVG; descriptive curves, not new inference."""
    budgets = sorted({r['budget'] for r in table})
    maximum_budget = max(budgets, default=.2)
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
                '<rect width="900" height="520" fill="white"/>',
                f'<text x="60" y="25" font-size="18">{html.escape(population)}: macro recall</text>']
    for tick in range(6):
        y = 420 - tick*72
        elements.append(f'<path d="M60 {y} H570" stroke="#ddd"/><text x="20" y="{y}">{tick/5:.1f}</text>')
    for i, policy in enumerate(sorted({r["policy"] for r in table})):
        rows = [r for r in table if r["policy"] == policy and r["recall"] is not None]
        # One panel per task is generated by the caller.
        points = " ".join(f'{60+r["budget"]/maximum_budget*500},{420-r["recall"]*360}' for r in sorted(rows, key=lambda r: r['budget']))
        color = ("#2563eb", "#d97706", "#7c3aed", "#166534", "#444", "#be123c")[i % 6]
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        elements.append(f'<text x="590" y="{70+i*30}" fill="{color}" font-size="11">{html.escape(policy)}</text>')
    grid = ', '.join(f'{100*b:g}%' for b in budgets)
    elements.append(f'<text x="70" y="465">Nominal inspection fraction (per-response ceiling): {grid}</text>')
    elements.append('<text x="70" y="490">Descriptive curves; statistical inference requires the underlying cluster-level analysis.</text></svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def run(config_path, data_dir, output, mode):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != VERSION:
        raise ValueError("Unsupported reproduction specification")
    if output.exists():
        raise ValueError("Output already exists: choose a new directory; historical artifacts are never overwritten")
    populations = ingest(config, data_dir)
    output.mkdir(parents=True)
    receipt = dict(schema_version=VERSION, mode=mode, config_sha256=sha(config_path),
                   api_calls=0, neural_inference=False, populations={}, status="RUNNING")
    dump(output / "receipt.json", receipt)
    try:
        for name, items, tasks, cluster_key in populations:
            destination = output / name; destination.mkdir()
            traces = []
            if mode == "historical":
                cfg = config.get("historical", {}).get(name)
                if not cfg:
                    raise ValueError(f"No authorized frozen trace specified for {name}")
                source = checked_file(data_dir, cfg["path"], cfg["sha256"])
                lookup = {r.response_id: r for r in items}; task_map = {t.name: t for t in tasks}
                seen = set()
                for line in source.read_text(encoding="utf-8").splitlines():
                    if not line.strip(): continue
                    row = json.loads(line)
                    key = (row["response_id"], row["task"], row["policy"], row["requested_budget"])
                    if key in seen or row["policy"] not in cfg["policies"]:
                        raise ValueError("Duplicate or unexpected historical policy/trace")
                    if row["requested_budget"] not in config["budgets"]:
                        raise ValueError("Unexpected historical budget")
                    seen.add(key)
                    traces.append(audit_trace(row, lookup[row["response_id"]], task_map[row["task"]], config["seed"]))
                expected = {(r.response_id, t.name, p, b) for r in items for t in tasks for p in cfg["policies"] for b in config["budgets"]}
                if seen != expected:
                    raise ValueError("Incomplete historical replay grid")
            else:
                policies = (RandomPolicy(), CitationScarcityPolicy(), LexicalCheckworthinessPolicy(), PrivilegedOraclePolicy())
                for response in items:
                    for task in tasks:
                        for policy in policies:
                            # FEVEROUS has no legitimate citation-count feature.
                            if (name.startswith("feverous_") or name == "synthetic") and isinstance(policy, CitationScarcityPolicy): continue
                            for budget in config["budgets"]:
                                traces.append(replay(response, policy, task, budget, seed=config["seed"]).to_dict())
            table, clusters = summarize(traces, cluster_key)
            csv_write(destination / "results.csv", table); csv_write(destination / "cluster_results.csv", clusters)
            for task in tasks:
                figure(destination / f"{task.name}.svg", [r for r in table if r["task"] == task.name], name)
            with (destination / "replay_traces.jsonl").open("w", encoding="utf-8") as handle:
                for row in traces: handle.write(json.dumps(row, sort_keys=True) + "\n")
            receipt["populations"][name] = dict(responses=len(items), units=sum(len(r.units) for r in items), traces=len(traces))
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt.update(status="FAIL", error=str(exc)); raise
    finally:
        receipt["outputs"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "receipt.json"}
        dump(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("deterministic", "historical"), default="deterministic")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.data_dir, args.output, args.mode), indent=2))


if __name__ == "__main__":
    main()
