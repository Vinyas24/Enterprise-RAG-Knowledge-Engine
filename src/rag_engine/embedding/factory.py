"""Provider selection.

One lookup table keeps conditionals out of the worker. Adding a provider means
registering it here and extending the configuration literal; the pipeline only
sees :class:`Embedder`.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.core.config import Settings

from .base import Embedder, EmbeddingProvider
from .batching import BatchedEmbedder
from .deterministic import DeterministicEmbeddingProvider
from .openai import OpenAIEmbeddingProvider


def _deterministic(settings: Settings) -> EmbeddingProvider:
    return DeterministicEmbeddingProvider(dimensions=settings.embedding_dimensions)


def _openai(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_api_key is None:
        raise ValueError("embedding_api_key is required")
    return OpenAIEmbeddingProvider(
        api_key=settings.embedding_api_key.get_secret_value(),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        base_url=settings.embedding_api_base,
        timeout_seconds=settings.embedding_timeout_seconds,
    )


_PROVIDERS: dict[str, Callable[[Settings], EmbeddingProvider]] = {
    "deterministic": _deterministic,
    "openai": _openai,
}


def build_embedder(settings: Settings, *, wrap_cache: bool = False) -> Embedder:
    """Construct the batching embedder for the configured provider."""
    builder = _PROVIDERS.get(settings.embedding_provider)
    if builder is None:
        # Settings validates the literal; this only fires for a caller that
        # bypassed validation with ``model_copy(update=...)``.
        raise ValueError(f"unknown embedding provider: {settings.embedding_provider!r}")
    embedder: Embedder = BatchedEmbedder(
        builder(settings),
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
        retry_base_seconds=settings.embedding_retry_base_seconds,
        retry_max_seconds=settings.embedding_retry_max_seconds,
    )
    if wrap_cache and settings.cache_enabled and settings.cache_embedding_enabled:
        from rag_engine.cache.embedding import CachedEmbedder

        embedder = CachedEmbedder(
            embedder,
            max_size=settings.cache_embedding_max_size,
            ttl_seconds=settings.cache_embedding_ttl_seconds,
        )
    return embedder
