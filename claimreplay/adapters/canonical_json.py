"""Adapter for ClaimReplay's canonical JSON format and user-created fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from ..io import response_from_dict
from ..models import FrozenResponse
from .base import DatasetAdapter


class CanonicalJsonAdapter(DatasetAdapter):
    dataset_name = "claimreplay_canonical_json"

    def load(self, path: str | Path, *, limit: int | None = None) -> list[FrozenResponse]:
        source = Path(path)
        if source.suffix.lower() == ".jsonl":
            payloads = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            payloads = payload if isinstance(payload, list) else [payload]
        if limit is not None:
            payloads = payloads[:limit]
        return [response_from_dict(payload) for payload in payloads]

