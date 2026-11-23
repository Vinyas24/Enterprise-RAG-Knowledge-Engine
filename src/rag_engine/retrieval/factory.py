"""Retriever factory."""

from __future__ import annotations

from rag_engine.core.config import Settings
from rag_engine.core.db import DatabaseProtocol
from rag_engine.embedding import Embedder
from rag_engine.vectorstore import VectorStore

from .base import Retriever
from .dense import DenseRetriever


def build_retriever(
    settings: Settings,
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    database: DatabaseProtocol,
) -> Retriever:
    """Build the configured retriever."""
    return DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
        database=database,
    )
