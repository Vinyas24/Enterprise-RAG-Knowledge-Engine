"""Vector-store contract.

Storage is separated from retrieval policy: this interface knows about vectors,
namespaces and exact-match metadata, not about access-control rules. Retrieval
(later phase) composes those rules into a filter before calling ``query``.

Namespaces are the isolation boundary. Each tenant gets its own namespace, so a
missing filter cannot leak another tenant's vectors. Metadata still carries
``tenant_id`` because defence in depth is cheap and future indexes may use one
shared namespace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Metadata = Mapping[str, str | int | float | bool]


def metadata_for_vector(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Drop ``None`` values, which vector-store metadata does not accept."""
    return {key: value for key, value in metadata.items() if value is not None}


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One vector and the metadata needed to cite it."""

    id: str
    values: list[float]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """One similarity-search result."""

    id: str
    score: float
    metadata: Metadata = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Swappable vector backend."""

    async def ensure_ready(self) -> None:
        """Create the index if needed and establish the data-plane client.

        Idempotent: the worker and the API both call it lazily.
        """
        ...

    async def upsert(self, namespace: str, records: Sequence[VectorRecord]) -> None:
        """Insert or replace vectors. Same id means replacement, not duplication."""
        ...

    async def delete(self, namespace: str, ids: Sequence[str]) -> None:
        """Delete explicit ids. Missing ids are not an error."""
        ...

    async def list_ids(self, namespace: str, *, prefix: str = "") -> list[str]:
        """List ids, optionally filtered by prefix.

        Used to purge a document's vectors without relying on filtered deletes,
        which not every Pinecone tier supports.
        """
        ...

    async def query(
        self,
        namespace: str,
        vector: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[VectorMatch]:
        """Cosine/similarity search with exact-match metadata predicates."""
        ...

    async def close(self) -> None:
        """Release resources. Must be idempotent."""
        ...
