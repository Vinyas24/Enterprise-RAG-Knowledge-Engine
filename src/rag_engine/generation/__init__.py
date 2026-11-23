"""Generation slice: prompt assembly, LLM calls and citation mapping."""

from __future__ import annotations

from .base import Citation, GenerationResult, LLMProvider
from .citations import extract_citations
from .deterministic import DeterministicLLMProvider
from .errors import GenerationError, LLMResponseError, LLMUnavailable
from .factory import build_llm, build_query_engine
from .openai import OpenAILLMProvider
from .prompts import NOT_FOUND_RESPONSE, assemble_prompt
from .service import QueryEngine

__all__ = [
    "NOT_FOUND_RESPONSE",
    "Citation",
    "DeterministicLLMProvider",
    "GenerationError",
    "GenerationResult",
    "LLMProvider",
    "LLMResponseError",
    "LLMUnavailable",
    "OpenAILLMProvider",
    "QueryEngine",
    "assemble_prompt",
    "build_llm",
    "build_query_engine",
    "extract_citations",
]
