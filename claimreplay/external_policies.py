"""Local, gold-free transfer adapter for the released AFaCTA classifier.

This does not reproduce the paper's LLM annotator or train a classifier.
Heavy dependencies are imported only when an uncached prediction is needed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from .models import PolicyView
from .policies import RankedUnit, TriageCost, VerificationPolicy

MODEL_REPO = "JingweiNi/roberta-base-afacta"
MODEL_REVISION = "5db3f67d68e384d486a08d26e0c180a483efa164"
AUTHOR_COMMIT = "25e8f3b8f9f3a6581867f59ba7e48f1085610f5d"
ADAPTER_VERSION = "afacta-transfer-v1.0.0"
POLICY_NAME = "external:afacta-verifiability-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def afacta_inputs(view: PolicyView) -> list[dict]:
    """Match author context concatenation without splitting frozen units.

    Author load_train/load_test use prev + sep + target + sep + next, omit
    absent boundary neighbors. For a singleton, use its target only (author
    loader assumes at least two sentences and would otherwise index past end).
    """
    if not isinstance(view, PolicyView):
        raise TypeError("AFaCTA accepts only an ordinary gold-free PolicyView")
    ids = [u.unit_id for u in view.units]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate unit IDs")
    if any(not isinstance(u.text, str) or not u.text.strip() for u in view.units):
        raise ValueError("Empty or malformed text")
    rows = []
    for i, unit in enumerate(view.units):
        previous = view.units[i - 1].text if i else None
        following = view.units[i + 1].text if i + 1 < len(view.units) else None
        parts = ([previous] if previous is not None else []) + [unit.text]
        if following is not None:
            parts.append(following)
        text = "</s>".join(parts)
        start = len(previous) + 4 if previous is not None else 0
        rows.append({"unit_id": unit.unit_id, "text": text,
                     "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "target_span": [start, start + len(unit.text)],
                     "previous_context": previous is not None,
                     "next_context": following is not None,
                     "singleton": len(view.units) == 1})
    return rows


class LocalAfactaScorer:
    """Pinned actual weights, native softmax class 1, CPU evaluation only."""

    def __init__(self, local_path: Path, acquisition: dict, *, batch_size=8, threads=4, seed=47):
        import importlib.metadata
        self.local_path = Path(local_path)
        self.acquisition = acquisition
        self.batch_size, self.threads, self.seed = batch_size, threads, seed
        if acquisition["repo"] != MODEL_REPO or acquisition["revision"] != MODEL_REVISION:
            raise ValueError("Wrong checkpoint identity")
        for name, info in acquisition["files"].items():
            if hashlib.sha256((self.local_path / name).read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
        self.identity = {"repo": MODEL_REPO, "revision": MODEL_REVISION,
                         "adapter": ADAPTER_VERSION, "positive_class": 1,
                         "max_length": 512, "truncation_side": "right",
                         "batch_size": batch_size, "threads": threads, "seed": seed,
                         "weights_sha256": acquisition["files"]["model.safetensors"]["sha256"],
                         "dependencies": {p: importlib.metadata.version(p) for p in
                                          ("torch", "transformers", "tokenizers", "numpy", "safetensors")}}
        self.model = None
        self.tokenizer = None
        self.loading_info = None
        self.load_seconds = 0.0
        self.forward_seconds = 0.0
        self.model_calls = 0
        self.executed_tokens = 0
        self.executed_padded_tokens = 0

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        started = time.perf_counter()
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        if self.tokenizer.sep_token != "</s>" or self.tokenizer.truncation_side != "right":
            raise ValueError("Unexpected native separator/truncation")
        self.model, self.loading_info = AutoModelForSequenceClassification.from_pretrained(
            self.local_path, local_files_only=True, trust_remote_code=False,
            output_loading_info=True, weights_only=True)
        if any(self.loading_info.values()):
            raise ValueError(f"Inexact weight loading: {self.loading_info}")
        if self.model.config.num_labels != 2:
            raise ValueError("Unexpected classifier dimensions")
        self.model.to("cpu").eval()
        self.load_seconds += time.perf_counter() - started

    def score(self, inputs: list[dict]) -> list[dict]:
        self._load()
        import torch
        results = []
        for offset in range(0, len(inputs), self.batch_size):
            chunk = inputs[offset:offset + self.batch_size]
            texts = [r["text"] for r in chunk]
            full = self.tokenizer(texts, truncation=False, padding=False, return_offsets_mapping=True)
            batch = self.tokenizer(texts, truncation=True, max_length=512, padding=True,
                                   return_offsets_mapping=True, return_tensors="pt")
            offsets = batch.pop("offset_mapping").tolist()
            count = batch["attention_mask"].sum(dim=1).tolist()
            started = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(**batch).logits
                probabilities = logits.softmax(dim=-1)
            self.forward_seconds += time.perf_counter() - started
            if not torch.isfinite(probabilities).all():
                raise ValueError("Nonfinite classifier output")
            self.model_calls += 1
            self.executed_tokens += sum(count)
            self.executed_padded_tokens += int(batch["input_ids"].numel())
            for i, item in enumerate(chunk):
                start, end = item["target_span"]
                overlaps = lambda span: span[1] > start and span[0] < end and span[1] > span[0]
                results.append({"score": float(probabilities[i, 1]),
                                "probabilities": probabilities[i].tolist(),
                                "untruncated_tokens": len(full["input_ids"][i]),
                                "input_tokens": count[i],
                                "truncated": len(full["input_ids"][i]) > 512,
                                "target_tokens": sum(overlaps(s) for s in full["offset_mapping"][i]),
                                "retained_target_tokens": sum(overlaps(s) for s in offsets[i])})
        return results


class AfactaPolicy(VerificationPolicy):
    name = POLICY_NAME

    def __init__(self, scorer, *, cache_path: Path | None = None):
        self.scorer, self.cache_path = scorer, cache_path
        self.identity = scorer.identity
        self.entries = {}
        if cache_path is not None and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload["identity"] != self.identity or payload.get("contains_gold") is not False:
                raise ValueError("Cache identity/schema mismatch")
            self.entries = payload["entries"]
        self.initial_entries = len(self.entries)
        self.last_records = []
        self.last_cost = None
        self.hits = self.misses = 0

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed  # This frozen classifier does not randomize rank.
        inputs = afacta_inputs(response)
        keys = [digest({"identity": self.identity, "input_sha256": r["input_sha256"],
                        "target_span": r["target_span"]}) for r in inputs]
        pending = {}
        hits = 0
        for key, row in zip(keys, inputs):
            if key in self.entries or key in pending:
                hits += 1
            else:
                pending[key] = row
        before_calls = getattr(self.scorer, "model_calls", 0)
        if pending:
            values = self.scorer.score(list(pending.values()))
            if len(values) != len(pending):
                raise ValueError("Scorer returned wrong row count")
            self.entries.update(zip(pending, values))
        records = []
        for key, row in zip(keys, inputs):
            value = self.entries[key]
            if not math.isfinite(value["score"]) or not 0 <= value["score"] <= 1:
                raise ValueError("Invalid native class probability")
            records.append({k: row[k] for k in row if k not in {"text", "target_span"}} |
                           {"cache_key": key, **value})
        self.last_records = records
        self.last_cost = TriageCost(units_considered=len(inputs), score_evaluations=len(inputs),
                                    model_calls=getattr(self.scorer, "model_calls", 0) - before_calls,
                                    input_tokens=sum(r["input_tokens"] for r in records),
                                    cache_hits=hits, cache_misses=len(pending))
        self.hits += hits
        self.misses += len(pending)
        return tuple(sorted((RankedUnit(r["unit_id"], r["score"]) for r in records),
                            key=lambda r: (-r.score, r.unit_id)))

    def triage_cost(self, response: PolicyView) -> TriageCost:
        if self.last_cost is None or len(response.units) != self.last_cost.units_considered:
            raise ValueError("Rank must precede cost")
        return self.last_cost

    def save(self):
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".partial")
            temporary.write_text(json.dumps({"identity": self.identity, "contains_gold": False,
                                            "entries": self.entries}, sort_keys=True), encoding="utf-8")
            temporary.replace(self.cache_path)


class RecordedAfactaPolicy(VerificationPolicy):
    """Replay already scored inputs, validating current gold-free content hashes."""
    name = POLICY_NAME

    def __init__(self, records: list[dict]):
        self.records = {}
        for row in records:
            key = digest([(r["unit_id"], r["input_sha256"]) for r in row["units"]])
            if key in self.records and row["ranking"] != self.records[key]["ranking"]:
                raise ValueError("Equivalent inputs have inconsistent cached rankings")
            self.records[key] = row

    def _record(self, response):
        key = digest([(r["unit_id"], r["input_sha256"]) for r in afacta_inputs(response)])
        if key not in self.records:
            raise ValueError("Scored content does not match frozen units")
        return self.records[key]

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        row = self._record(response)
        inputs = afacta_inputs(response)
        expected = [(r["unit_id"], r["input_sha256"]) for r in inputs]
        observed = [(r["unit_id"], r["input_sha256"]) for r in row["units"]]
        if expected != observed:
            raise ValueError("Scored content does not match frozen units")
        if row["model_revision"] != MODEL_REVISION:
            raise ValueError("Recorded model revision mismatch")
        return tuple(sorted((RankedUnit(r["unit_id"], r["score"]) for r in row["units"]),
                            key=lambda r: (-r.score, r.unit_id)))

    def triage_cost(self, response: PolicyView) -> TriageCost:
        return TriageCost(**self._record(response)["triage_cost"])
