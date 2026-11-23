"""Vector-store selection."""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.core.config import Settings

from .base import VectorStore
from .memory import MemoryVectorStore
from .pinecone import PineconeVectorStore


def _memory(_settings: Settings) -> VectorStore:
    return MemoryVectorStore()


def _pinecone(settings: Settings) -> VectorStore:
    if settings.pinecone_api_key is None:
        raise ValueError("pinecone_api_key is required")
    return PineconeVectorStore(
        api_key=settings.pinecone_api_key.get_secret_value(),
        index_name=settings.pinecone_index_name,
        dimension=settings.embedding_dimensions,
        metric=settings.pinecone_metric,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        control_host=settings.pinecone_control_host,
        index_host=settings.pinecone_index_host,
        upsert_batch_size=settings.vector_upsert_batch_size,
        delete_batch_size=settings.vector_delete_batch_size,
        list_page_size=settings.vector_list_page_size,
        timeout_seconds=settings.pinecone_timeout_seconds,
    )


_STORES: dict[str, Callable[[Settings], VectorStore]] = {
    "memory": _memory,
    "pinecone": _pinecone,
}


def build_vector_store(settings: Settings) -> VectorStore:
    """Construct the store named by ``settings.vector_store``."""
    builder = _STORES.get(settings.vector_store)
    if builder is None:
        raise ValueError(f"unknown vector store: {settings.vector_store!r}")
    return builder(settings)
