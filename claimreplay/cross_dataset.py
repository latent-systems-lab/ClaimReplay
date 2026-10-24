"""Deterministic controlled review queues for native benchmark claims.

Queue construction takes IDs only. Native assessments and dependency metadata
are evaluator-only. SciFact evidence comes from source citations, never rationales.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .adapters.feverous import FeverousAdapter
from .adapters.scifact import SciFactAdapter
from .models import (EvidenceBundle, EvidenceItem, FrozenAttributes, FrozenResponse,
                     GoldAssessment, Provenance, SupportLevel, VerificationUnit)
from .policies import RankedUnit, TriageCost, VerificationPolicy, EvidenceAwareVerificationPolicy
from .tasks import EvaluationTask

VERSION = "controlled-claim-review-queues-v1"
AF_POLICY = "transfer:afacta-target-only-v1"
NLI_POLICY = "transfer:nli-first-cited-abstract-v1"
CITATION_POLICY = "transfer:citation-scarcity-source-documents-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def ordered_ids(ids, dataset, split, seed=47, cap=None):
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs")
    return sorted(ids, key=lambda i: (digest([VERSION, dataset, split, seed, i]), i))[:cap]


def partition(ids, size):
    if size < 2:
        raise ValueError("Queue size must permit ranking")
    return [list(ids[i:i + size]) for i in range(0, len(ids), size)]


def task(dataset, target):
    return EvaluationTask(f"{dataset}_{target}", "ReviewNeeded",
                          frozenset({SupportLevel.NO_SUPPORT}))


def map_label(label, target):
    if label in {"SUPPORT", "SUPPORTS"}:
        return SupportLevel.FULL_SUPPORT
    if label in {"REFUTES", "CONTRADICT"}:
        return SupportLevel.NO_SUPPORT
    if label not in {"NOT_ENOUGH_INFO", "NOT ENOUGH INFO"}:
        raise ValueError(f"Unknown native label: {label}")
    if target == "native_non_support":
        return SupportLevel.NO_SUPPORT
    if target == "native_refutation":
        return SupportLevel.UNASSESSED
    raise ValueError(f"Unknown target: {target}")


def load_sources(root, spec):
    """Reuse existing ingestion; replace gold-rationale evidence on the policy side."""
    f = spec["datasets"]["feverous"]
    fever = FeverousAdapter(split=f["split"]).load_claims(root / f["claims"])
    s = spec["datasets"]["scifact"]
    sci = SciFactAdapter(root / s["corpus"], split=s["split"]).load_claims(root / s["claims"])
    corpus = {str(r["doc_id"]): r for r in read_jsonl(root / s["corpus"])}
    result = {}
    for dataset, claims in (("feverous", fever), ("scifact", sci)):
        cfg = spec["datasets"][dataset]
        by_id = {c.claim_id: c for c in claims}
        selected = ordered_ids(by_id, dataset, cfg["split"], spec["seed"], cfg["cap"])
        records = []
        for source_id in selected:
            c = by_id[source_id]
            citations = c.provenance.citation_ids if dataset == "scifact" else ()
            # Preserve citation-list order, attach full abstracts to each unit
            # separately. Never aggregate unrelated queue claims into context.
            evidence = tuple(EvidenceItem(d, " ".join(corpus[d]["abstract"])) for d in citations)
            records.append({"unit_id": "benchmark:" + digest([dataset, source_id])[:24],
                            "source_id": source_id, "text": c.text,
                            "label": c.assessment.raw_label,
                            "citations": list(citations), "evidence": EvidenceBundle(evidence) if evidence else None,
                            "dependencies": c.provenance.metadata.get("source_group_ids", [])})
        result[dataset] = {"records": records, "native_count": len(claims),
                           "native_labels": dict(Counter(c.assessment.raw_label for c in claims))}
    return result


def dependency_groups(records, largest_queue=40):
    """Conservative connected components: shared provenance/text OR shared quota.

Largest-queue quota edges give identical grouping across both size variants.
Evidence-derived FEVEROUS pages are used only for dependency analysis, not
queue membership, order, policy inputs, scores or eligibility.
"""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    seen = {}
    for i, r in enumerate(records):
        keys = ["source:" + str(v) for v in r["dependencies"]]
        keys += ["text:" + digest(r["text"]), "quota:" + str(i // largest_queue)]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    components = {}
    for i, r in enumerate(records):
        components.setdefault(find(i), []).append(r["unit_id"])
    groups = sorted(components.values(), key=lambda ids: min(ids))
    return {uid: f"dependency-{i:04d}" for i, ids in enumerate(groups) for uid in ids}


def responses(dataset, records, groups, size, target):
    output = []
    for q, chunk in enumerate(partition(records, size)):
        units = tuple(VerificationUnit(r["unit_id"], r["text"],
                      Provenance(dataset, r["source_id"]),
                      FrozenAttributes.from_mapping({"citation_count": len(r["citations"])}
                          if dataset == "scifact" else {}), r["evidence"]) for r in chunk)
        labels = tuple(GoldAssessment(r["unit_id"], map_label(r["label"], target), r["label"])
                       for r in chunk)
        group = {groups[r["unit_id"]] for r in chunk}
        if len(group) != 1:
            raise ValueError("Queue straddles dependency components")
        output.append(FrozenResponse(f"queue:{dataset}:n{size}:q{q:04d}", units, labels,
                      FrozenAttributes.from_mapping({"dataset": dataset, "group_id": group.pop(),
                        "setting": VERSION, "queue_size": size})))
    return output


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RecordedTextPolicy(VerificationPolicy):
    name = AF_POLICY
    def __init__(self, scores):
        self.scores = {r["unit_id"]: r for r in scores}
    def rank(self, view, *, seed):
        del seed
        rows = []
        for u in view.units:
            r = self.scores[u.unit_id]
            if r["text_sha256"] != hashlib.sha256(u.text.encode()).hexdigest():
                raise ValueError("Recorded score text mismatch")
            rows.append(RankedUnit(u.unit_id, r["afacta"]["score"]))
        return tuple(sorted(rows, key=lambda r: (-r.score, r.unit_id)))
    def triage_cost(self, view):
        return TriageCost(len(view.units), len(view.units), input_tokens=sum(
            self.scores[u.unit_id]["afacta"]["input_tokens"] for u in view.units), cache_hits=len(view.units))


class RecordedEvidencePolicy(EvidenceAwareVerificationPolicy):
    name = NLI_POLICY
    def __init__(self, scores):
        self.scores = {r["unit_id"]: r for r in scores if "nli" in r}
    def rank(self, view, *, seed):
        del seed
        rows = []
        for u in view.units:
            r = self.scores[u.unit_id]
            if digest([u.text, u.evidence_texts[0] if u.evidence_texts else None]) != r["nli_input_sha256"]:
                raise ValueError("Recorded evidence mismatch")
            rows.append(RankedUnit(u.unit_id, r["nli"]["risk"]))
        return tuple(sorted(rows, key=lambda r: (-r.score, r.unit_id)))
    def triage_cost(self, view):
        return TriageCost(len(view.units), len(view.units), input_tokens=sum(
            self.scores[u.unit_id]["nli"]["input_tokens"] for u in view.units), cache_hits=len(view.units))
