"""Provider-neutral structured LLM inference with resumable content caches.

The module deliberately knows nothing about evaluator assessments.  Callers
construct requests only from the information regime authorized for a policy or
judge.  Raw provider responses and validated parsed records are stored in
separate directories so cached experiments remain inspectable.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LlmInferenceError(RuntimeError):
    """A provider request failed without a valid structured result."""


class LlmBudgetExceeded(RuntimeError):
    """A predeclared request or token ceiling would be exceeded."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 30.0
    jitter_seed: int = 47

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff_seconds < 0 or self.maximum_backoff_seconds < 0:
            raise ValueError("retry delays cannot be negative")


@dataclass(frozen=True)
class ProviderConfiguration:
    provider: str
    model: str
    endpoint: str = "https://api.openai.com/v1/responses"
    reasoning_effort: str = "none"
    max_output_tokens: int = 32
    timeout_seconds: float = 60.0
    store: bool = False
    api_key_environment_variable: str = "OPENAI_API_KEY"
    implementation_version: str = "claimreplay-structured-llm-v1"

    def identity(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "store": self.store,
            "implementation_version": self.implementation_version,
        }


@dataclass(frozen=True)
class StructuredRequest:
    request_id: str
    task: str
    system_prompt: str
    user_payload: Mapping[str, Any]
    schema_name: str
    json_schema: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def content_identity(self) -> dict[str, Any]:
        # request_id and metadata are provenance only and may contain benchmark
        # identifiers. They never enter the provider prompt or cache identity.
        return {
            "task": self.task,
            "system_prompt": self.system_prompt,
            "user_payload": dict(self.user_payload),
            "schema_name": self.schema_name,
            "json_schema": dict(self.json_schema),
        }

    @property
    def input_hash(self) -> str:
        return content_hash(self.content_identity())

    def provider_payload(self, config: ProviderConfiguration) -> dict[str, Any]:
        return {
            "model": config.model,
            "store": config.store,
            "reasoning": {"effort": config.reasoning_effort},
            "max_output_tokens": config.max_output_tokens,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": canonical_json(dict(self.user_payload))}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "strict": True,
                    "schema": dict(self.json_schema),
                }
            },
        }


@dataclass(frozen=True)
class ProviderResult:
    raw_response: Mapping[str, Any]
    parsed: Mapping[str, Any]
    input_tokens: int | None
    output_tokens: int | None
    response_id: str | None = None
    attempts: int = 1


@dataclass(frozen=True)
class InferenceRecord:
    request_id: str
    request_hash: str
    cache_key: str
    status: str
    parsed: Mapping[str, Any] | None
    raw_response_path: str | None
    parsed_response_path: str | None
    input_tokens: int | None
    output_tokens: int | None
    response_id: str | None
    attempts: int
    cache_hit: bool
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "cache_key": self.cache_key,
            "status": self.status,
            "parsed": dict(self.parsed) if self.parsed is not None else None,
            "raw_response_path": self.raw_response_path,
            "parsed_response_path": self.parsed_response_path,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "response_id": self.response_id,
            "attempts": self.attempts,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class StructuredLlmProvider(Protocol):
    configuration: ProviderConfiguration

    def complete(self, request: StructuredRequest) -> ProviderResult:
        ...


def _extract_response_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for output in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        if not isinstance(output, Mapping):
            continue
        for item in output.get("content", []) if isinstance(output.get("content"), list) else []:
            if isinstance(item, Mapping) and item.get("type") == "output_text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return text
            if isinstance(item, Mapping) and item.get("type") == "refusal":
                raise LlmInferenceError(f"Provider refusal: {item.get('refusal', 'unspecified')}")
    raise LlmInferenceError("Provider response contains no output_text")


class OpenAIResponsesProvider:
    """Small dependency-free client for OpenAI Responses structured outputs."""

    def __init__(
        self,
        configuration: ProviderConfiguration,
        *,
        retry_policy: RetryPolicy | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if configuration.provider != "openai":
            raise ValueError("OpenAIResponsesProvider requires provider='openai'")
        self.configuration = configuration
        self.retry_policy = retry_policy or RetryPolicy()
        self._opener = opener or urllib.request.urlopen

    def credential_available(self) -> bool:
        return bool(os.environ.get(self.configuration.api_key_environment_variable, "").strip())

    def complete(self, request: StructuredRequest) -> ProviderResult:
        api_key = os.environ.get(self.configuration.api_key_environment_variable, "").strip()
        if not api_key:
            raise LlmInferenceError(
                f"Missing credential environment variable {self.configuration.api_key_environment_variable}"
            )
        body = canonical_json(request.provider_payload(self.configuration)).encode("utf-8")
        rng = random.Random(self.retry_policy.jitter_seed + int(request.input_hash[:8], 16))
        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            http_request = urllib.request.Request(
                self.configuration.endpoint,
                data=body,
                method="POST",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            try:
                with self._opener(http_request, timeout=self.configuration.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = _extract_response_text(payload)
                parsed = json.loads(text)
                usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
                return ProviderResult(
                    raw_response=payload,
                    parsed=parsed,
                    input_tokens=_optional_int(usage.get("input_tokens")),
                    output_tokens=_optional_int(usage.get("output_tokens")),
                    response_id=str(payload.get("id")) if payload.get("id") else None,
                    attempts=attempt,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = LlmInferenceError(f"HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, LlmInferenceError) as exc:
                last_error = exc
            if attempt < self.retry_policy.max_attempts:
                delay = min(
                    self.retry_policy.maximum_backoff_seconds,
                    self.retry_policy.initial_backoff_seconds * (2 ** (attempt - 1)),
                )
                time.sleep(delay * (0.75 + 0.5 * rng.random()))
        raise LlmInferenceError(f"Request failed after {self.retry_policy.max_attempts} attempts: {last_error}")


class MockStructuredProvider:
    """Deterministic provider used for schema, cache and isolation tests."""

    def __init__(
        self,
        responder: Callable[[StructuredRequest], Mapping[str, Any]],
        *,
        model: str = "deterministic-mock-v1",
    ) -> None:
        self.configuration = ProviderConfiguration(provider="mock", model=model, endpoint="mock://local")
        self.responder = responder
        self.calls = 0

    def complete(self, request: StructuredRequest) -> ProviderResult:
        self.calls += 1
        parsed = dict(self.responder(request))
        raw = {"id": f"mock-{request.input_hash[:16]}", "output_text": canonical_json(parsed),
               "usage": {"input_tokens": estimate_request_tokens(request), "output_tokens": 8}}
        return ProviderResult(raw, parsed, raw["usage"]["input_tokens"], 8, raw["id"], 1)


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def estimate_request_tokens(request: StructuredRequest) -> int:
    # Conservative tokenizer-independent planning estimate. Actual provider
    # usage replaces it in execution receipts.
    characters = len(request.system_prompt) + len(canonical_json(dict(request.user_payload)))
    return math.ceil(characters / 3) + 32


class StructuredResponseCache:
    def __init__(self, root: Path, configuration: ProviderConfiguration) -> None:
        self.root = Path(root)
        self.configuration = configuration
        self.raw_dir = self.root / "raw"
        self.parsed_dir = self.root / "parsed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    def key(self, request: StructuredRequest) -> str:
        return content_hash({"provider": self.configuration.identity(), "request": request.content_identity()})

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.raw_dir / f"{key}.json", self.parsed_dir / f"{key}.json"

    def get(self, request: StructuredRequest) -> InferenceRecord | None:
        key = self.key(request)
        raw_path, parsed_path = self._paths(key)
        if raw_path.exists() != parsed_path.exists():
            raise LlmInferenceError(f"Truncated cache entry for {key}")
        if not parsed_path.exists():
            return None
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        parsed_record = json.loads(parsed_path.read_text(encoding="utf-8"))
        expected = {
            "cache_key": key,
            "request_hash": request.input_hash,
            "provider_identity": self.configuration.identity(),
        }
        for name, value in expected.items():
            if parsed_record.get(name) != value:
                raise LlmInferenceError(f"Cache identity mismatch for {key}: {name}")
        if content_hash(raw) != parsed_record.get("raw_sha256"):
            raise LlmInferenceError(f"Raw response hash mismatch for {key}")
        return InferenceRecord(
            request_id=request.request_id,
            request_hash=request.input_hash,
            cache_key=key,
            status="ok",
            parsed=parsed_record["parsed"],
            raw_response_path=raw_path.as_posix(),
            parsed_response_path=parsed_path.as_posix(),
            input_tokens=parsed_record.get("input_tokens"),
            output_tokens=parsed_record.get("output_tokens"),
            response_id=parsed_record.get("response_id"),
            attempts=int(parsed_record.get("attempts", 1)),
            cache_hit=True,
            metadata=request.metadata,
        )

    def put(self, request: StructuredRequest, result: ProviderResult) -> InferenceRecord:
        key = self.key(request)
        raw_path, parsed_path = self._paths(key)
        parsed_record = {
            "schema_version": "claimreplay.structured-response-cache.v1",
            "cache_key": key,
            "request_hash": request.input_hash,
            "provider_identity": self.configuration.identity(),
            "raw_sha256": content_hash(result.raw_response),
            "parsed": dict(result.parsed),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "response_id": result.response_id,
            "attempts": result.attempts,
        }
        _atomic_json(raw_path, result.raw_response)
        _atomic_json(parsed_path, parsed_record)
        return InferenceRecord(
            request_id=request.request_id,
            request_hash=request.input_hash,
            cache_key=key,
            status="ok",
            parsed=result.parsed,
            raw_response_path=raw_path.as_posix(),
            parsed_response_path=parsed_path.as_posix(),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            response_id=result.response_id,
            attempts=result.attempts,
            cache_hit=False,
            metadata=request.metadata,
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class ExecutionLimits:
    maximum_requests: int
    maximum_planned_tokens: int
    concurrency: int = 4

    def __post_init__(self) -> None:
        if self.maximum_requests < 1 or self.maximum_planned_tokens < 1 or self.concurrency < 1:
            raise ValueError("Execution limits must be positive")


def execute_structured_requests(
    requests: Sequence[StructuredRequest],
    provider: StructuredLlmProvider,
    cache: StructuredResponseCache,
    limits: ExecutionLimits,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[InferenceRecord, ...]:
    """Execute unique content requests and reconstruct the original order."""

    unique: dict[str, StructuredRequest] = {}
    for request in requests:
        unique.setdefault(cache.key(request), request)
    planned_tokens = sum(estimate_request_tokens(request) + provider.configuration.max_output_tokens
                         for request in unique.values())
    if len(unique) > limits.maximum_requests:
        raise LlmBudgetExceeded(f"Planned {len(unique)} requests exceeds limit {limits.maximum_requests}")
    if planned_tokens > limits.maximum_planned_tokens:
        raise LlmBudgetExceeded(
            f"Planned token ceiling {planned_tokens} exceeds limit {limits.maximum_planned_tokens}"
        )

    by_key: dict[str, InferenceRecord] = {}
    pending: list[tuple[str, StructuredRequest]] = []
    for key, request in unique.items():
        cached = cache.get(request)
        if cached is None:
            pending.append((key, request))
        else:
            validator(cached.parsed or {})
            by_key[key] = cached

    def execute(item: tuple[str, StructuredRequest]) -> tuple[str, InferenceRecord]:
        key, request = item
        try:
            result = provider.complete(request)
            validated = validator(result.parsed)
            result = ProviderResult(result.raw_response, validated, result.input_tokens,
                                    result.output_tokens, result.response_id, result.attempts)
            return key, cache.put(request, result)
        except Exception as exc:  # each failure remains explicit and resumable
            return key, InferenceRecord(
                request_id=request.request_id,
                request_hash=request.input_hash,
                cache_key=key,
                status="failed",
                parsed=None,
                raw_response_path=None,
                parsed_response_path=None,
                input_tokens=None,
                output_tokens=None,
                response_id=None,
                attempts=getattr(provider, "retry_policy", RetryPolicy()).max_attempts,
                cache_hit=False,
                error=f"{type(exc).__name__}: {exc}",
                metadata=request.metadata,
            )

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=limits.concurrency) as executor:
            for key, record in executor.map(execute, pending):
                by_key[key] = record

    return tuple(
        InferenceRecord(**{**by_key[cache.key(request)].__dict__, "request_id": request.request_id,
                           "metadata": request.metadata})
        for request in requests
    )
