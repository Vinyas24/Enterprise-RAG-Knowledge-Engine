"""Retrieval slice: dense retrieval and candidate search."""

from __future__ import annotations

from .base import RetrievedChunk, Retriever
from .dense import DenseRetriever
from .errors import RetrievalError
from .factory import build_retriever

__all__ = [
    "DenseRetriever",
    "RetrievalError",
    "RetrievedChunk",
    "Retriever",
    "build_retriever",
]
