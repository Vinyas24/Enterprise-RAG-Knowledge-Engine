"""Generation error taxonomy."""

from __future__ import annotations


class GenerationError(Exception):
    """Base exception for generation failures."""


class LLMUnavailable(GenerationError):
    """The language model backend is unreachable or timed out."""


class LLMResponseError(GenerationError):
    """The language model backend returned an unexpected or malformed response."""
