"""Re-ranker selection.

One lookup table keeps provider conditionals out of the API and the evaluation
harness. ``off`` returns ``None`` so callers can keep the Phase 6 retriever
untouched and skip composition entirely.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.core.config import RerankProvider, Settings

from .base import Reranker
from .cohere import DEFAULT_COHERE_MODEL, CohereReranker
from .cross_encoder import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from .lexical import LexicalReranker


def _cohere(settings: Settings) -> Reranker:
    if settings.rerank_api_key is None:
        raise ValueError("rerank_api_key is required when rerank_provider is 'cohere'")
    return CohereReranker(
        api_key=settings.rerank_api_key.get_secret_value(),
        model=settings.rerank_model or DEFAULT_COHERE_MODEL,
        base_url=settings.rerank_api_base,
        timeout_seconds=settings.rerank_timeout_seconds,
    )


def _cross_encoder(settings: Settings) -> Reranker:
    return CrossEncoderReranker(
        model_name=settings.rerank_model or DEFAULT_CROSS_ENCODER_MODEL,
        batch_size=settings.rerank_batch_size,
        device=settings.rerank_device,
    )


def _lexical(settings: Settings) -> Reranker:
    del settings
    return LexicalReranker()


_BUILDERS: dict[RerankProvider, Callable[[Settings], Reranker]] = {
    RerankProvider.COHERE: _cohere,
    RerankProvider.CROSS_ENCODER: _cross_encoder,
    RerankProvider.LEXICAL: _lexical,
}


def build_reranker(settings: Settings) -> Reranker | None:
    """Construct the configured re-ranker, or ``None`` when disabled."""
    if settings.rerank_provider is RerankProvider.OFF:
        return None
    builder = _BUILDERS.get(settings.rerank_provider)
    if builder is None:  # pragma: no cover - Settings validates the enum
        raise ValueError(f"unknown rerank provider: {settings.rerank_provider!r}")
    return builder(settings)


def resolve_rerank_model(settings: Settings) -> str | None:
    """The model name that will actually be used, for reporting.

    ``lexical`` has no model. Reporting an ignored ``rerank_model`` there would
    suggest a model-shaped run when the provider is pure BM25.
    """
    if settings.rerank_provider in (RerankProvider.OFF, RerankProvider.LEXICAL):
        return None
    if settings.rerank_model:
        return settings.rerank_model
    if settings.rerank_provider is RerankProvider.COHERE:
        return DEFAULT_COHERE_MODEL
    return DEFAULT_CROSS_ENCODER_MODEL
