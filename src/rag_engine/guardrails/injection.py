"""Direct prompt injection and jailbreak detection for user queries.

Identifies adversarial attempts to hijack the generation model, exfiltrate
system prompts, break out of context boundaries, or override instructions.
"""

from __future__ import annotations

import re

from .base import InjectionDetectionResult

# Categorized injection signatures
_SYSTEM_OVERRIDE_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|forget|override|negate|bypass)\s+"
        r"(?:all\s+)?(?:previous|prior|above|existing|system)\s+"
        r"(?:instructions|prompts|rules|directives|commands|constraints)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:stop\s+following|do\s+not\s+follow)\s+"
        r"(?:the\s+)?(?:system|previous|above)\s+"
        r"(?:instructions|prompts|rules|commands)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfrom\s+now\s+on\b.*?\b(?:you\s+must|you\s+shall|only\s+respond|always\s+say)\b",
        re.IGNORECASE,
    ),
]

_JAILBREAK_PATTERNS = [
    re.compile(
        r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+"
        r"(?:DAN|AIM|unrestricted|jailbroken|evil\s+bot|an\s+unfiltered)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:enter|enable|switch\s+to)\s+"
        r"(?:developer\s+mode|jailbreak\s+mode|unrestricted\s+mode|god\s+mode)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdo\s+anything\s+now\b",
        re.IGNORECASE,
    ),
]

_EXFILTRATION_PATTERNS = [
    re.compile(
        r"\b(?:reveal|show|print|display|output|repeat|state|leak)\s+"
        r"(?:your\s+|the\s+)?(?:system\s+prompt|initial\s+prompt|initial\s+instructions|"
        r"system\s+instructions|internal\s+instructions|secret\s+prompt|"
        r"developer\s+instructions|developer\s+message)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:are|were)\s+your\s+(?:system|initial|original|secret)\s+"
        r"(?:instructions|prompts|directives)\b",
        re.IGNORECASE,
    ),
]

_DELIMITER_BREAKOUT_PATTERNS = [
    re.compile(r"</?(?:untrusted_)?document_context>", re.IGNORECASE),
    re.compile(r"</?context>", re.IGNORECASE),
    re.compile(r"<\|im_start\|>|<\|im_end\|>"),
    re.compile(r"\[/?INST\]"),
    re.compile(r"```+\s*system", re.IGNORECASE),
]


def detect_injection(query: str) -> InjectionDetectionResult:
    """Scan user query for prompt injection and jailbreak signatures."""
    clean_query = query.strip()
    if not clean_query:
        return InjectionDetectionResult(is_injection=False)

    matched_patterns: list[str] = []

    # 1. System override checks
    for pattern in _SYSTEM_OVERRIDE_PATTERNS:
        if pattern.search(clean_query):
            matched_patterns.append(f"system_override:{pattern.pattern}")

    # 2. Jailbreak persona checks
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(clean_query):
            matched_patterns.append(f"jailbreak:{pattern.pattern}")

    # 3. System prompt exfiltration checks
    for pattern in _EXFILTRATION_PATTERNS:
        if pattern.search(clean_query):
            matched_patterns.append(f"exfiltration:{pattern.pattern}")

    # 4. Context delimiter breakout checks
    for pattern in _DELIMITER_BREAKOUT_PATTERNS:
        if pattern.search(clean_query):
            matched_patterns.append(f"delimiter_breakout:{pattern.pattern}")

    if matched_patterns:
        confidence = min(1.0, 0.7 + 0.15 * len(matched_patterns))
        return InjectionDetectionResult(
            is_injection=True,
            matched_patterns=matched_patterns,
            confidence=confidence,
            reason=f"Matched {len(matched_patterns)} adversarial prompt injection patterns",
        )

    return InjectionDetectionResult(is_injection=False, confidence=0.0)
