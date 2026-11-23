"""Embedding providers and the batching wrapper.

Callers depend on :class:`Embedder`; concrete providers are selected by
:func:`build_embedder` from configuration and never imported by the pipeline.
"""

from __future__ import annotations

from .base import Embedder, EmbeddingProvider
from .batching import BatchedEmbedder
from .deterministic import DeterministicEmbeddingProvider
from .errors import (
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTransientError,
)
from .factory import build_embedder
from .openai import OpenAIEmbeddingProvider

__all__ = [
    "BatchedEmbedder",
    "DeterministicEmbeddingProvider",
    "Embedder",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingRateLimitError",
    "EmbeddingResponseError",
    "EmbeddingTransientError",
    "OpenAIEmbeddingProvider",
    "build_embedder",
]
