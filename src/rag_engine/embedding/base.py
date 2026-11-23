"""Embedding contracts.

``EmbeddingProvider`` is the raw provider: one call, no batching or retry
policy. ``Embedder`` is what the ingestion pipeline depends on: batching and
rate-limit handling are implementation details behind it. Keeping the two
separate means a provider is trivial to fake in tests, while retry behaviour is
tested once instead of once per provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A raw text-to-vector backend."""

    name: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one batch. Must return one vector per input, in order."""
        ...

    async def close(self) -> None:
        """Release network resources. Must be idempotent."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """The pipeline-facing embedding interface."""

    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed any number of texts, preserving input order."""
        ...

    async def close(self) -> None:
        """Release provider resources."""
        ...
