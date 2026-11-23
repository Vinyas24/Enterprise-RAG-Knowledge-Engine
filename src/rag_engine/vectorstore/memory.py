"""In-memory vector store: the default for tests and local runs.

It implements exactly the same contract as the Pinecone adapter, including
namespaces, prefix listing, upserts by id and metadata filtering. Integration
tests therefore exercise the real pipeline wiring without a network dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .base import Metadata, VectorMatch, VectorRecord


class MemoryVectorStore:
    def __init__(self, *, dimension: int | None = None) -> None:
        self._dimension = dimension
        self._namespaces: dict[str, dict[str, VectorRecord]] = {}
        self._closed = False

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, namespace: str, records: Sequence[VectorRecord]) -> None:
        for record in records:
            if self._dimension is not None and len(record.values) != self._dimension:
                raise ValueError(
                    f"vector {record.id!r} has {len(record.values)} dimensions, "
                    f"expected {self._dimension}"
                )
        bucket = self._namespaces.setdefault(namespace, {})
        for record in records:
            bucket[record.id] = VectorRecord(
                id=record.id,
                values=list(record.values),
                metadata=dict(record.metadata),
            )

    async def delete(self, namespace: str, ids: Sequence[str]) -> None:
        bucket = self._namespaces.get(namespace)
        if bucket is None:
            return
        for vector_id in ids:
            bucket.pop(vector_id, None)

    async def list_ids(self, namespace: str, *, prefix: str = "") -> list[str]:
        bucket = self._namespaces.get(namespace, {})
        return sorted(vector_id for vector_id in bucket if vector_id.startswith(prefix))

    async def query(
        self,
        namespace: str,
        vector: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[VectorMatch]:
        bucket = self._namespaces.get(namespace, {})
        norm = math.sqrt(sum(value * value for value in vector))
        matches: list[VectorMatch] = []
        for record in bucket.values():
            if not _matches(record.metadata, metadata_filter):
                continue
            score = _cosine(vector, norm, record.values)
            matches.append(VectorMatch(id=record.id, score=score, metadata=dict(record.metadata)))
        matches.sort(key=lambda match: (-match.score, match.id))
        return matches[:top_k]

    async def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def count(self, namespace: str) -> int:
        """Test/operator helper; not part of the protocol."""
        return len(self._namespaces.get(namespace, {}))


def _cosine(left: Sequence[float], left_norm: float, right: Sequence[float]) -> float:
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False)) / (left_norm * right_norm)


def _matches(metadata: Metadata, predicates: Mapping[str, str | int | float | bool] | None) -> bool:
    if not predicates:
        return True
    return all(metadata.get(key) == value for key, value in predicates.items())
