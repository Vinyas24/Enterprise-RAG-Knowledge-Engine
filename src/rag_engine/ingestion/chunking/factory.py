"""Strategy selection.

One lookup table instead of conditionals spread through the worker. Adding a
variant means registering it here and adding the name to the configuration
literal; the worker only ever sees :class:`ChunkingStrategy`.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.core.config import Settings

from .base import ChunkingStrategy
from .recursive import RecursiveChunker


def _recursive(settings: Settings) -> ChunkingStrategy:
    return RecursiveChunker(
        chunk_size=settings.chunk_size_chars,
        chunk_overlap=settings.chunk_overlap_chars,
    )


_STRATEGIES: dict[str, Callable[[Settings], ChunkingStrategy]] = {
    "recursive": _recursive,
}


def build_chunker(settings: Settings) -> ChunkingStrategy:
    """Construct the strategy named by ``settings.chunk_strategy``."""
    builder = _STRATEGIES.get(settings.chunk_strategy)
    if builder is None:
        # Settings validates the literal, so this only fires when a caller
        # bypasses validation (for example ``model_copy(update=...)``).
        raise ValueError(f"unknown chunking strategy: {settings.chunk_strategy!r}")
    return builder(settings)
