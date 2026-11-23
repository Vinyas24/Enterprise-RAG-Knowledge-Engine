"""Stable, cross-store identities for chunks.

A chunk has two identities that must agree:

* a PostgreSQL primary key (a UUID), and
* a vector id in the vector store (a string).

Both are derived from ``(document_id, chunk_index)`` rather than generated
randomly. Re-ingesting a document therefore writes the same ids again, which
turns upserts into replacements instead of duplicates. The vector id is also
prefixed with the document id so every vector of one document shares a prefix;
that lets deletion enumerate a document's vectors without a metadata filter
(Pinecone Local, and some production tiers, do not support filtered deletes).
"""

from __future__ import annotations

from uuid import UUID, uuid5

# A fixed namespace keeps ids reproducible across processes and releases. It
# is a name, not a secret; changing it would orphan every existing vector.
_CHUNK_NAMESPACE = UUID("5f0c3f0e-6f6f-5c3e-9a0e-2f6d6c1f7a10")

_VECTOR_SEPARATOR = ":"


def chunk_vector_id(document_id: UUID, chunk_index: int) -> str:
    """Vector-store id for one chunk of a document."""
    return f"{document_id}{_VECTOR_SEPARATOR}{chunk_index}"


def document_vector_prefix(document_id: UUID) -> str:
    """Prefix shared by every vector id of a document."""
    return f"{document_id}{_VECTOR_SEPARATOR}"


def chunk_id(document_id: UUID, chunk_index: int) -> UUID:
    """PostgreSQL id for one chunk, deterministic across re-ingestion."""
    return uuid5(_CHUNK_NAMESPACE, chunk_vector_id(document_id, chunk_index))
