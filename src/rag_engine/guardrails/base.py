"""Guardrails contracts, data structures, and actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rag_engine.generation.base import Citation


@dataclass(frozen=True, slots=True)
class PIIEntity:
    """A detected sensitive entity inside user input."""

    entity_type: str
    start: int
    end: int
    placeholder: str
    value: str


@dataclass(frozen=True, slots=True)
class PIIDetectionResult:
    """Outcome of PII detection and masking."""

    text: str
    entities: list[PIIEntity] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.entities)


@dataclass(frozen=True, slots=True)
class InjectionDetectionResult:
    """Outcome of direct or indirect prompt injection detection."""

    is_injection: bool
    matched_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class GroundednessResult:
    """Outcome of factual grounding verification against retrieved passages."""

    is_grounded: bool
    score: float
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """Outcome of citation verification against candidate passages."""

    is_valid: bool
    valid_citations: list[Citation] = field(default_factory=list)
    dropped_citations: list[int] = field(default_factory=list)


class GuardrailAction(StrEnum):
    """Actions taken by the input, retrieval, or output guardrail pipeline."""

    ALLOWED = "allowed"
    REFUSED_INJECTION = "refused_injection"
    REFUSED_UNSUPPORTED = "refused_unsupported"
    REFUSED_DOCUMENT_INJECTION = "refused_document_injection"
    RATE_LIMITED = "rate_limited"
    INPUT_LENGTH_EXCEEDED = "input_length_exceeded"
    WEAK_RETRIEVAL = "weak_retrieval"


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Overall outcome of a guardrail evaluation step."""

    passed: bool
    action: GuardrailAction
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
