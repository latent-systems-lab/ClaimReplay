"""Dataset adapters shipped with ClaimReplay."""

from .base import ClaimDatasetAdapter, DatasetAdapter
from .canonical_json import CanonicalJsonAdapter
from .feverous import FeverousAdapter
from .scifact import SciFactAdapter
from .trec_rag_2024 import TrecRag2024Adapter

__all__ = [
    "CanonicalJsonAdapter",
    "ClaimDatasetAdapter",
    "DatasetAdapter",
    "FeverousAdapter",
    "SciFactAdapter",
    "TrecRag2024Adapter",
]
