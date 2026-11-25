"""Retrieval slice: dense, keyword, hybrid retrieval and rank fusion."""

from __future__ import annotations

from .base import RetrievedChunk, Retriever
from .dense import DenseRetriever
from .errors import RetrievalError
from .factory import build_retriever
from .fusion import reciprocal_rank_fusion
from .hybrid import HybridRetriever
from .keyword import KeywordRetriever, bm25_score, selection_terms, tokenize

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "KeywordRetriever",
    "RetrievalError",
    "RetrievedChunk",
    "Retriever",
    "bm25_score",
    "build_retriever",
    "reciprocal_rank_fusion",
    "selection_terms",
    "tokenize",
]
