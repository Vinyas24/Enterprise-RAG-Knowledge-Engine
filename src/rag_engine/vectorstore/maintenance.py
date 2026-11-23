"""Cross-store cleanup helpers.

Whether or not a backend supports metadata-filtered deletes, a document's
vectors can always be enumerated by id prefix. These helpers keep the policy in
one place: the worker removes stale vectors after re-ingestion, and the delete
endpoint purges a document before deleting its row.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from rag_engine.core.ids import document_vector_prefix

from .base import VectorStore

_DELETE_CHUNK = 500


async def delete_vectors(
    store: VectorStore, namespace: str, ids: Iterable[str], *, chunk_size: int = _DELETE_CHUNK
) -> int:
    """Delete explicit ids in bounded batches. Returns the number requested."""
    pending: list[str] = []
    deleted = 0
    for vector_id in ids:
        pending.append(vector_id)
        if len(pending) == chunk_size:
            await store.delete(namespace, pending)
            deleted += len(pending)
            pending.clear()
    if pending:
        await store.delete(namespace, pending)
        deleted += len(pending)
    return deleted


async def delete_document_vectors(
    store: VectorStore,
    namespace: str,
    document_id: UUID,
    *,
    known_ids: Iterable[str] = (),
) -> int:
    """Remove every vector belonging to one document.

    ``known_ids`` lets the caller delete ids it read from PostgreSQL even when
    the backend's list API is eventually consistent; the prefix sweep then
    catches orphans left by a failed earlier attempt.
    """
    deleted = await delete_vectors(store, namespace, known_ids)
    prefix = document_vector_prefix(document_id)
    remaining = await store.list_ids(namespace, prefix=prefix)
    if remaining:
        await store.delete(namespace, remaining)
        deleted += len(remaining)
    return deleted


async def delete_stale_vectors(
    store: VectorStore,
    namespace: str,
    document_id: UUID,
    *,
    keep_ids: Iterable[str],
    previous_ids: Iterable[str] = (),
) -> int:
    """Delete vectors of ``document_id`` that are not in ``keep_ids``.

    ``previous_ids`` comes from the old chunk rows, which is immediate and
    robust. The prefix sweep catches vectors left behind by an attempt that
    wrote some batches and then failed before the database row was updated.
    """
    keep = set(keep_ids)
    stale = {vector_id for vector_id in previous_ids if vector_id not in keep}
    prefix = document_vector_prefix(document_id)
    stale.update(
        vector_id
        for vector_id in await store.list_ids(namespace, prefix=prefix)
        if vector_id not in keep
    )
    if not stale:
        return 0
    await store.delete(namespace, sorted(stale))
    return len(stale)
