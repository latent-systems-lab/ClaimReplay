"""Common adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import BenchmarkClaim, FrozenResponse


class DatasetAdapter(ABC):
    """Adapter for datasets that already contain complete responses."""

    dataset_name: str

    @abstractmethod
    def load(self, path: str | Path, *, limit: int | None = None) -> list[FrozenResponse]:
        raise NotImplementedError


class ClaimDatasetAdapter(ABC):
    """Adapter for labeled source claims used in controlled generation."""

    dataset_name: str

    @abstractmethod
    def load_claims(self, path: str | Path, *, limit: int | None = None) -> list[BenchmarkClaim]:
        raise NotImplementedError
