"""Document-level injection defense (indirect prompt injection).

Neutralizes malicious instructions hidden inside ingested files, ensuring
document content is treated strictly as passive data and never obeyed as
executable instructions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_engine.retrieval.base import RetrievedChunk

# Patterns that indicate an ingested chunk is attempting an indirect injection
_INDIRECT_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:important|urgent|system\s+update|admin\s+note|new\s+instruction):\s*"
        r"(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|other|above)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:instead\s+of\s+answering|do\s+not\s+answer)\s+the\s+question\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+answer\s+to\s+every\s+question\s+is|always\s+respond\s+with)\s+['\"]?([^'\"\n]+)['\"]?",
        re.IGNORECASE,
    ),
]

_DELIMITER_TAG_RE = re.compile(
    r"</?(?:untrusted_)?document_context[^>]*>",
    re.IGNORECASE,
)


def sanitize_chunk_content(content: str) -> str:
    """Neutralize boundary tags inside chunk content to prevent delimiter breakouts."""
    # Escape angle brackets in delimiter-like tags
    return _DELIMITER_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), content
    )


def has_indirect_injection(content: str) -> bool:
    """Detect whether a chunk contains adversarial indirect injection directives."""
    return any(p.search(content) for p in _INDIRECT_INJECTION_PATTERNS)


def extract_injected_canaries(content: str) -> list[str]:
    """Extract override strings that indirect injection payloads attempt to force."""
    canaries: list[str] = []
    for match in re.finditer(
        r"\b(?:say|output|respond\s+with)\s+['\"]?([A-Za-z0-9_-]{4,32})['\"]?",
        content,
        re.IGNORECASE,
    ):
        canaries.append(match.group(1).strip().lower())
    return canaries


def check_answer_for_injection_leakage(
    answer: str,
    passages: Sequence[RetrievedChunk],
) -> bool:
    """Check whether the generated answer obeyed an indirect injection command.

    Returns True if an injection payload was detected in the answer.
    """
    answer_lower = answer.lower()
    for chunk in passages:
        if has_indirect_injection(chunk.content):
            canaries = extract_injected_canaries(chunk.content)
            for canary in canaries:
                if canary in answer_lower:
                    return True
    return False
