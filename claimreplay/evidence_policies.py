"""Evidence-aware non-oracle policies and reproducible NLI score caching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from typing import Protocol

from .models import EvidencePolicyView
from .policies import EvidenceAwareVerificationPolicy, RankedUnit, TriageCost


DEFAULT_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
DEFAULT_MODEL_REVISION = "0233b6e1207aa367aa01ac84abd8efecc842f2d1"
CACHE_SCHEMA = "claimreplay.evidence-score-cache.v2"


def _cache_key(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class NliScore:
    contradiction: float
    entailment: float
    neutral: float
    input_tokens: int

    def __post_init__(self) -> None:
        probabilities = (self.contradiction, self.entailment, self.neutral)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
            raise ValueError("NLI probabilities must be finite and in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError("NLI probabilities must sum to one")
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")


@dataclass(frozen=True)
class NliBatchOutput:
    scores: tuple[NliScore, ...]
    model_calls: int


class NliScorer(Protocol):
    model_id: str
    revision: str

    @property
    def cache_identity(self) -> dict:
        """JSON configuration/runtime identity; must change when scoring changes."""

    def score_pairs(self, pairs: tuple[tuple[str, str], ...]) -> NliBatchOutput:
        """Score ``(evidence premise, response-unit hypothesis)`` pairs."""


class DeterministicMockNliScorer:
    """Offline contract fixture; never reported as an empirical NLI baseline."""

    model_id = "claimreplay/mock-nli"
    revision = "v1"

    @property
    def cache_identity(self) -> dict:
        return {"implementation": "claimreplay.mock-nli.v1", "backend": "pure-python"}

    def score_pairs(self, pairs: tuple[tuple[str, str], ...]) -> NliBatchOutput:
        scores = []
        for premise, hypothesis in pairs:
            tokens = len(premise.split()) + len(hypothesis.split())
            if hypothesis.casefold() in premise.casefold():
                scores.append(NliScore(0.01, 0.98, 0.01, tokens))
            else:
                scores.append(NliScore(0.10, 0.10, 0.80, tokens))
        return NliBatchOutput(tuple(scores), model_calls=1 if pairs else 0)


class TransformersNliScorer:
    """Pinned CPU inference backend; defaults to local-only model loading."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        batch_size: int = 8,
        local_files_only: bool = True,
        asset_sha256: dict[str, str] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        asset_sha256 = dict(asset_sha256 or {})
        if any(
            not isinstance(name, str) or not name
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
            for name, digest in asset_sha256.items()
        ):
            raise ValueError("asset_sha256 must map nonempty names to 64-character SHA-256 hex digests")
        self.model_id, self.revision = model_id, revision
        self.batch_size, self.local_files_only = batch_size, local_files_only
        self.asset_sha256 = dict(sorted((name, digest.lower()) for name, digest in asset_sha256.items()))
        self._tokenizer = self._model = self._torch = None

    @property
    def cache_identity(self) -> dict:
        # Importing torch reads runtime settings, but loads no model/weights and
        # runs no inference. It is needed to distinguish externally set threads
        # and dtype. Missing dependencies cannot produce a real cache entry.
        try:
            import torch
        except ImportError:
            torch_runtime = {"available": False}
        else:
            torch_runtime = {
                "available": True, "version": str(torch.__version__),
                "threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "default_dtype": str(torch.get_default_dtype()),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }
        packages = {}
        for name in ("torch", "transformers", "tokenizers", "sentencepiece", "protobuf", "safetensors", "numpy"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "implementation": "claimreplay.transformers-nli.v2",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "batch_size": self.batch_size, "device": "cpu", "seed": 0,
            "deterministic_algorithms": True, "model_eval": True,
            "tokenizer": {"model_id": self.model_id, "revision": self.revision,
                "use_fast": True, "max_length": 512, "truncation": "longest_first",
                "truncation_side": "right", "padding": "longest",
                "pair_order": ["evidence", "claim"]},
            "asset_sha256": self.asset_sha256,
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "packages": packages, "torch_runtime": torch_runtime,
            "environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "ONEDNN_MAX_CPU_ISA",
                "MKL_CBWR", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")},
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install ClaimReplay's evidence optional dependencies") from error
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=self.local_files_only,
            use_fast=True, truncation_side="right",
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=self.local_files_only
        ).to("cpu")
        self._model.eval()
        self._torch = torch

    def score_pairs(self, pairs: tuple[tuple[str, str], ...]) -> NliBatchOutput:
        if not pairs:
            return NliBatchOutput((), 0)
        self._load()
        label_map = {str(value).lower(): int(key) for key, value in self._model.config.id2label.items()}
        required = {"contradiction", "entailment", "neutral"}
        if not required <= set(label_map):
            raise ValueError(f"Model labels do not expose {sorted(required)}")
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            encoded = self._tokenizer(
                [row[0] for row in batch], [row[1] for row in batch],
                padding=True, truncation="longest_first", max_length=512, return_tensors="pt",
            )
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
            lengths = encoded["attention_mask"].sum(dim=1).cpu().tolist()
            for row, tokens in zip(probabilities, lengths):
                scores.append(NliScore(
                    contradiction=float(row[label_map["contradiction"]]),
                    entailment=float(row[label_map["entailment"]]),
                    neutral=float(row[label_map["neutral"]]),
                    input_tokens=int(tokens),
                ))
        return NliBatchOutput(tuple(scores), model_calls=math.ceil(len(pairs) / self.batch_size))


class EvidenceScoreCache:
    """Persistent gold-free cache keyed only by model/config and input hashes."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.entries: dict[str, dict] = {}
        if self.path and self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("Evidence score cache is unreadable or truncated") from error
            if payload.get("schema_version") != CACHE_SCHEMA or payload.get("contains_gold") is not False:
                raise ValueError("Unsupported evidence score cache format; v2 required. Keep older caches separate and rebuild using the current configuration.")
            if not isinstance(payload.get("entries"), dict):
                raise ValueError("Evidence cache entries must be an object")
            for key, value in payload["entries"].items():
                self.set(key, value)

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def set(self, key: str, value: dict) -> None:
        allowed = {"model_id", "revision", "method", "claim_sha256", "evidence_sha256", "execution_identity", "score"}
        if set(value) != allowed:
            raise ValueError("Evidence cache entry has unexpected fields")
        fields = {name: item for name, item in value.items() if name != "score"}
        if not isinstance(value["execution_identity"], dict) or not value["execution_identity"]:
            raise ValueError("Evidence cache requires a nonempty execution identity")
        if _cache_key(fields) != key:
            raise ValueError("Evidence cache key does not match its input/configuration metadata")
        NliScore(**value["score"])
        self.entries[key] = value

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": CACHE_SCHEMA, "contains_gold": False, "entries": self.entries}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


class NliEvidenceRiskPolicy(EvidenceAwareVerificationPolicy):
    """Rank by ``1 - entailment`` for the first cited evidence segment.

    This is a deterministic prioritization score, not a calibrated probability
    of factual error. Missing evidence receives the fixed maximum risk score.
    """

    name = "baseline:evidence-nli-risk-v2"
    method = "one_minus_entailment:first_citation:missing_max"

    def __init__(self, scorer: NliScorer, *, cache: EvidenceScoreCache | None = None):
        self.scorer = scorer
        self.cache = cache or EvidenceScoreCache()
        self._costs: dict[str, TriageCost] = {}

    def _key(self, claim: str, evidence: str, *, execution_identity: dict | None = None) -> tuple[str, dict]:
        claim_hash = hashlib.sha256(claim.encode("utf-8")).hexdigest()
        evidence_hash = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
        fields = {
            "model_id": self.scorer.model_id,
            "revision": self.scorer.revision,
            "method": self.method,
            "claim_sha256": claim_hash,
            "evidence_sha256": evidence_hash,
            "execution_identity": self.scorer.cache_identity if execution_identity is None else execution_identity,
        }
        key = _cache_key(fields)
        return key, fields

    def rank(self, response: EvidencePolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        scores: dict[str, float] = {}
        pending = []
        cache_hits = cache_misses = evidence_characters = input_tokens = 0
        logical_evaluations = 0
        execution_identity = self.scorer.cache_identity
        for unit in response.units:
            if not unit.evidence_texts or not unit.evidence_texts[0]:
                scores[unit.unit_id] = 1.0
                continue
            evidence = unit.evidence_texts[0]
            evidence_characters += len(evidence)
            logical_evaluations += 1
            key, fields = self._key(unit.text, evidence, execution_identity=execution_identity)
            cached = self.cache.get(key)
            if cached is not None:
                if {name: value for name, value in cached.items() if name != "score"} != fields:
                    raise ValueError("Evidence cache input/configuration mismatch")
                score = NliScore(**cached["score"])
                scores[unit.unit_id] = 1.0 - score.entailment
                input_tokens += score.input_tokens
                cache_hits += 1
            else:
                pending.append((unit.unit_id, key, fields, evidence, unit.text))
                cache_misses += 1
        output = self.scorer.score_pairs(tuple((row[3], row[4]) for row in pending))
        if len(output.scores) != len(pending):
            raise ValueError("NLI scorer returned the wrong number of scores")
        for pending_row, score in zip(pending, output.scores):
            unit_id, key, fields, _, _ = pending_row
            scores[unit_id] = 1.0 - score.entailment
            input_tokens += score.input_tokens
            self.cache.set(key, {**fields, "score": score.__dict__.copy()})
        self.cache.save()
        self._costs[response.response_hash] = TriageCost(
            units_considered=len(response.units),
            score_evaluations=logical_evaluations,
            model_calls=output.model_calls,
            input_tokens=input_tokens,
            evidence_characters=evidence_characters,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
        return tuple(sorted(
            (RankedUnit(unit.unit_id, scores[unit.unit_id]) for unit in response.units),
            key=lambda row: (-float(row.score), row.unit_id),
        ))

    def triage_cost(self, response: EvidencePolicyView) -> TriageCost:
        if response.response_hash not in self._costs:
            raise RuntimeError("rank must be called before triage_cost")
        return self._costs[response.response_hash]
