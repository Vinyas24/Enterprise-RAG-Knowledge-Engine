"""LLM provider and QueryEngine factory."""

from __future__ import annotations

from rag_engine.core.config import Settings
from rag_engine.retrieval.base import Retriever

from .base import LLMProvider
from .deterministic import DeterministicLLMProvider
from .openai import OpenAILLMProvider
from .service import QueryEngine


def build_llm(settings: Settings) -> LLMProvider:
    """Build the configured LLM provider."""
    if settings.llm_provider == "deterministic":
        return DeterministicLLMProvider()
    if settings.llm_provider == "openai":
        if settings.llm_api_key is None:
            raise ValueError("llm_api_key is required")
        return OpenAILLMProvider(
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            base_url=settings.llm_api_base,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    raise ValueError(f"unknown LLM provider: {settings.llm_provider!r}")


def build_query_engine(
    settings: Settings,
    *,
    retriever: Retriever,
    llm: LLMProvider,
) -> QueryEngine:
    """Build the query engine coordinator."""
    return QueryEngine(
        retriever=retriever,
        llm=llm,
        settings=settings,
    )
