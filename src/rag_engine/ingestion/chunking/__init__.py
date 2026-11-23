"""Chunking strategies and the factory that selects one.

The package boundary is the interface: callers ask for a strategy by name and
receive something that implements :class:`ChunkingStrategy`. They never import
a concrete splitter directly, so a variant can be swapped or benchmarked
without touching the worker.
"""

from __future__ import annotations

from .base import ChunkDraft, ChunkingStrategy
from .factory import build_chunker
from .recursive import RecursiveChunker, normalise_blocks, split_text, split_text_spans

__all__ = [
    "ChunkDraft",
    "ChunkingStrategy",
    "RecursiveChunker",
    "build_chunker",
    "normalise_blocks",
    "split_text",
    "split_text_spans",
]
