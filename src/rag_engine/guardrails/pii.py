"""PII detection and masking for user queries and audit records.

Redacts sensitive information (emails, phone numbers, SSNs, credit cards,
API keys) before input reaches vector search, LLM prompts, or application logs.
"""

from __future__ import annotations

import re

from .base import PIIDetectionResult, PIIEntity

# Standard RFC-compliant email pattern
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Phone numbers (US & international formats: +1..., (555) 000-0000, 555-000-0000)
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# US Social Security Number: XXX-XX-XXXX
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit card patterns (Visa, Mastercard, Amex, Discover, 13-19 digits formatted or raw)
_CREDIT_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d{4}[- ]){3}\d{4}\b|\b\d{13,19}\b")

# API keys and secret bearer tokens
_API_KEY_PATTERNS = [
    re.compile(r"\b(?:sk|pk)-(?:live|test|proj-)?[a-zA-Z0-9_-]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b[Bb]earer\s+([A-Za-z0-9_\-\.]{25,})\b"),
]


def luhn_checksum_valid(number_str: str) -> bool:
    """Verify numeric string against Luhn checksum (MOD-10)."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def mask_pii(text: str) -> PIIDetectionResult:
    """Detect and redact PII entities, returning masked text and entity details."""
    entities: list[PIIEntity] = []

    # 1. API keys & bearer tokens
    for pattern in _API_KEY_PATTERNS:
        for match in pattern.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="api_key",
                    start=match.start(),
                    end=match.end(),
                    placeholder="[API_KEY]",
                    value=match.group(0),
                )
            )

    # 2. Email addresses
    for match in _EMAIL_RE.finditer(text):
        entities.append(
            PIIEntity(
                entity_type="email",
                start=match.start(),
                end=match.end(),
                placeholder="[EMAIL_ADDRESS]",
                value=match.group(0),
            )
        )

    # 3. Social Security Numbers
    for match in _SSN_RE.finditer(text):
        entities.append(
            PIIEntity(
                entity_type="ssn",
                start=match.start(),
                end=match.end(),
                placeholder="[SSN]",
                value=match.group(0),
            )
        )

    # 4. Credit cards with Luhn verification
    for match in _CREDIT_CARD_CANDIDATE_RE.finditer(text):
        raw = match.group(0)
        digits_only = re.sub(r"\D", "", raw)
        if luhn_checksum_valid(digits_only):
            entities.append(
                PIIEntity(
                    entity_type="credit_card",
                    start=match.start(),
                    end=match.end(),
                    placeholder="[CREDIT_CARD]",
                    value=raw,
                )
            )

    # 5. Phone numbers (exclude matches that overlap with SSN or Credit Card)
    for match in _PHONE_RE.finditer(text):
        start, end = match.start(), match.end()
        # Ensure no overlap with existing detected entities
        if not any(e.start <= start < e.end or e.start < end <= e.end for e in entities):
            entities.append(
                PIIEntity(
                    entity_type="phone",
                    start=start,
                    end=end,
                    placeholder="[PHONE_NUMBER]",
                    value=match.group(0),
                )
            )

    if not entities:
        return PIIDetectionResult(text=text, entities=[])

    # Sort entities in reverse order of start position to replace from back to front
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

    # Filter out overlapping entities (keep the earliest / longer match)
    non_overlapping: list[PIIEntity] = []
    last_start = len(text) + 1
    for entity in sorted_entities:
        if entity.end <= last_start:
            non_overlapping.append(entity)
            last_start = entity.start

    masked_text = text
    for entity in non_overlapping:
        masked_text = masked_text[: entity.start] + entity.placeholder + masked_text[entity.end :]

    # Return entities in ascending order of original appearance
    return PIIDetectionResult(
        text=masked_text, entities=sorted(non_overlapping, key=lambda e: e.start)
    )
