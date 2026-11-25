"""Retriever factory."""

from __future__ import annotations

from rag_engine.core.config import RetrievalMode, Settings
from rag_engine.core.db import DatabaseProtocol
from rag_engine.embedding import Embedder
from rag_engine.vectorstore import VectorStore

from .base import Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .keyword import KeywordRetriever


def build_retriever(
    settings: Settings,
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    database: DatabaseProtocol,
) -> Retriever:
    """Build the retriever selected by ``retrieval_mode``.

    Dense-only mode returns the Phase 4 retriever untouched, so the saved
    baseline stays reproducible from the same factory.
    """
    dense = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
        database=database,
    )
    if settings.retrieval_mode is RetrievalMode.DENSE:
        return dense

    return HybridRetriever(
        dense=dense,
        keyword=KeywordRetriever(
            database=database,
            candidate_limit=settings.retrieval_keyword_candidates,
        ),
        dense_weight=settings.hybrid_dense_weight,
        keyword_weight=settings.hybrid_keyword_weight,
        rrf_k=settings.hybrid_rrf_k,
    )
