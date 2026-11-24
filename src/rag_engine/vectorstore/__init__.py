"""Vector storage behind one interface.

The worker and API depend on :class:`VectorStore`; the concrete backend is
selected by :func:`build_vector_store`. Tests use the in-memory implementation,
Compose uses Pinecone Local, and production uses a Pinecone index with the same
code path.
"""

from __future__ import annotations

from .base import Metadata, VectorMatch, VectorRecord, VectorStore, metadata_for_vector
from .errors import VectorStoreConfigurationError, VectorStoreError, VectorStoreUnavailable
from .factory import build_vector_store
from .maintenance import delete_document_vectors, delete_stale_vectors, delete_vectors
from .memory import MemoryVectorStore
from .pinecone import PineconeVectorStore

__all__ = [
    "MemoryVectorStore",
    "Metadata",
    "PineconeVectorStore",
    "VectorMatch",
    "VectorRecord",
    "VectorStore",
    "VectorStoreConfigurationError",
    "VectorStoreError",
    "VectorStoreUnavailable",
    "build_vector_store",
    "delete_document_vectors",
    "delete_stale_vectors",
    "delete_vectors",
    "metadata_for_vector",
]
