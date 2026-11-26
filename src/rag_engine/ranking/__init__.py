"""Ranking slice: re-ranking of merged retrieval candidates.

The re-ranker is an interface so a hosted provider (Cohere), a local
cross-encoder, and the deterministic offline provider are interchangeable.
"""

from __future__ import annotations

from .base import RerankAttempt, Reranker
from .cohere import CohereReranker
from .cross_encoder import CrossEncoderReranker
from .errors import (
    RerankError,
    RerankResponseError,
    RerankTransientError,
    RerankUnavailableError,
)
from .factory import build_reranker, resolve_rerank_model
from .lexical import LexicalReranker
from .reranked import RerankedRetriever

__all__ = [
    "CohereReranker",
    "CrossEncoderReranker",
    "LexicalReranker",
    "RerankAttempt",
    "RerankError",
    "RerankResponseError",
    "RerankTransientError",
    "RerankUnavailableError",
    "RerankedRetriever",
    "Reranker",
    "build_reranker",
    "resolve_rerank_model",
]
