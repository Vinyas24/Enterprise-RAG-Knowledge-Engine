"""Guardrails service: input safety, indirect defense, groundedness, and auditing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from uuid import UUID

from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.generation.base import Citation
from rag_engine.generation.prompts import NOT_FOUND_RESPONSE
from rag_engine.retrieval.base import RetrievedChunk

from .audit import AuditLogger, AuditRecord, hash_query
from .base import GuardrailAction, GuardrailResult
from .citation_verifier import validate_citations
from .document_injection import (
    check_answer_for_injection_leakage,
    sanitize_chunk_content,
)
from .groundedness import check_groundedness
from .injection import detect_injection
from .pii import mask_pii
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

INJECTION_REFUSAL_RESPONSE = (
    "I cannot answer this question because it violates input safety policies."
)


class GuardrailsService:
    """End-to-end security and safety enforcement for the RAG query pipeline."""

    def __init__(
        self,
        *,
        settings: Settings,
        audit_logger: AuditLogger | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._settings = settings
        self._audit_logger = audit_logger
        self._rate_limiter = rate_limiter or RateLimiter()

    async def check_input(
        self,
        *,
        tenant_id: UUID,
        query: str,
        clearance: AccessLevel,
        request_id: str | None = None,
    ) -> tuple[str, GuardrailResult]:
        """Validate, rate-limit, inspect for injection, and mask PII on input query."""
        if not self._settings.guardrails_enabled:
            return query, GuardrailResult(passed=True, action=GuardrailAction.ALLOWED)

        clean_query = query.strip()

        # 1. Rate limiting
        allowed, _remaining, retry_after = await self._rate_limiter.is_allowed(
            key=f"tenant:{tenant_id}",
            limit=self._settings.guardrails_rate_limit_per_minute,
        )
        if not allowed:
            res = GuardrailResult(
                passed=False,
                action=GuardrailAction.RATE_LIMITED,
                reason=(
                    f"Rate limit of {self._settings.guardrails_rate_limit_per_minute} "
                    "req/min exceeded"
                ),
                details={"retry_after_seconds": retry_after},
            )
            await self._record_audit(
                tenant_id=tenant_id,
                raw_query=clean_query,
                processed_query=clean_query,
                clearance=clearance,
                answer="Rate limit exceeded.",
                action=GuardrailAction.RATE_LIMITED,
                details=res.details,
                request_id=request_id,
            )
            return clean_query, res

        # 2. Length validation
        if len(clean_query) > self._settings.guardrails_max_query_length:
            res = GuardrailResult(
                passed=False,
                action=GuardrailAction.INPUT_LENGTH_EXCEEDED,
                reason=(
                    f"Query length {len(clean_query)} exceeds maximum "
                    f"{self._settings.guardrails_max_query_length}"
                ),
                details={
                    "length": len(clean_query),
                    "max_length": self._settings.guardrails_max_query_length,
                },
            )
            await self._record_audit(
                tenant_id=tenant_id,
                raw_query=clean_query,
                processed_query=clean_query,
                clearance=clearance,
                answer="Query length exceeded.",
                action=GuardrailAction.INPUT_LENGTH_EXCEEDED,
                details=res.details,
                request_id=request_id,
            )
            return clean_query, res

        # 3. Prompt injection detection
        if self._settings.guardrails_detect_prompt_injection:
            injection_res = detect_injection(clean_query)
            if injection_res.is_injection:
                res = GuardrailResult(
                    passed=False,
                    action=GuardrailAction.REFUSED_INJECTION,
                    reason=injection_res.reason,
                    details={
                        "matched_patterns": injection_res.matched_patterns,
                        "confidence": injection_res.confidence,
                    },
                )
                await self._record_audit(
                    tenant_id=tenant_id,
                    raw_query=clean_query,
                    processed_query=clean_query,
                    clearance=clearance,
                    answer=INJECTION_REFUSAL_RESPONSE,
                    action=GuardrailAction.REFUSED_INJECTION,
                    details=res.details,
                    request_id=request_id,
                )
                return clean_query, res

        # 4. PII detection and masking
        processed_query = clean_query
        pii_details: dict[str, Any] = {}
        if self._settings.guardrails_mask_pii:
            pii_res = mask_pii(clean_query)
            processed_query = pii_res.text
            if pii_res.has_pii:
                pii_details["pii_types"] = [e.entity_type for e in pii_res.entities]
                pii_details["pii_count"] = len(pii_res.entities)

        return processed_query, GuardrailResult(
            passed=True, action=GuardrailAction.ALLOWED, details=pii_details
        )

    def sanitize_passages(
        self,
        passages: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Neutralize boundary tags in retrieved chunks to prevent indirect injection."""
        return [replace(chunk, content=sanitize_chunk_content(chunk.content)) for chunk in passages]

    async def check_output(
        self,
        *,
        tenant_id: UUID,
        query: str,
        answer: str,
        passages: Sequence[RetrievedChunk],
        citations: Sequence[Citation],
        clearance: AccessLevel,
        request_id: str | None = None,
        latency_ms: float = 0.0,
    ) -> tuple[str, list[Citation], bool, GuardrailResult]:
        """Validate output for groundedness, citation integrity, and indirect injection."""
        if not self._settings.guardrails_enabled:
            return (
                answer,
                list(citations),
                False,
                GuardrailResult(passed=True, action=GuardrailAction.ALLOWED),
            )

        # 1. Check for indirect document injection leakage in answer
        if check_answer_for_injection_leakage(answer, passages):
            logger.warning("guardrails.indirect_injection_leakage_detected")
            res = GuardrailResult(
                passed=False,
                action=GuardrailAction.REFUSED_DOCUMENT_INJECTION,
                reason="Generated answer compromised by indirect document injection",
            )
            await self._record_audit(
                tenant_id=tenant_id,
                raw_query=query,
                processed_query=query,
                clearance=clearance,
                answer=NOT_FOUND_RESPONSE,
                action=GuardrailAction.REFUSED_DOCUMENT_INJECTION,
                details=res.details,
                retrieved_chunks=passages,
                citations=[],
                latency_ms=latency_ms,
                request_id=request_id,
            )
            return NOT_FOUND_RESPONSE, [], True, res

        # 2. Citation verification
        valid_citations = list(citations)
        if self._settings.guardrails_validate_citations:
            cit_res = validate_citations(answer, passages, citations)
            valid_citations = cit_res.valid_citations

        # 3. Groundedness check
        if self._settings.guardrails_check_groundedness:
            ground_res = check_groundedness(
                answer=answer,
                passages=passages,
                min_score=self._settings.guardrails_min_groundedness_score,
            )
            if not ground_res.is_grounded:
                logger.info(
                    "guardrails.ungrounded_answer_refused",
                    extra={"groundedness_score": round(ground_res.score, 3)},
                )
                res = GuardrailResult(
                    passed=False,
                    action=GuardrailAction.REFUSED_UNSUPPORTED,
                    reason=f"Answer ungrounded in context (score {round(ground_res.score, 3)})",
                    details={
                        "groundedness_score": ground_res.score,
                        "unsupported_claims": ground_res.unsupported_claims,
                    },
                )
                await self._record_audit(
                    tenant_id=tenant_id,
                    raw_query=query,
                    processed_query=query,
                    clearance=clearance,
                    answer=NOT_FOUND_RESPONSE,
                    action=GuardrailAction.REFUSED_UNSUPPORTED,
                    details=res.details,
                    retrieved_chunks=passages,
                    citations=[],
                    latency_ms=latency_ms,
                    request_id=request_id,
                )
                return NOT_FOUND_RESPONSE, [], True, res

        # 4. Success / Allowed
        res = GuardrailResult(passed=True, action=GuardrailAction.ALLOWED)
        await self._record_audit(
            tenant_id=tenant_id,
            raw_query=query,
            processed_query=query,
            clearance=clearance,
            answer=answer,
            action=GuardrailAction.ALLOWED,
            details=res.details,
            retrieved_chunks=passages,
            citations=valid_citations,
            latency_ms=latency_ms,
            request_id=request_id,
        )
        return answer, valid_citations, False, res

    async def _record_audit(
        self,
        *,
        tenant_id: UUID,
        raw_query: str,
        processed_query: str,
        clearance: AccessLevel,
        answer: str,
        action: GuardrailAction,
        details: dict[str, Any],
        retrieved_chunks: Sequence[RetrievedChunk] = (),
        citations: Sequence[Citation] = (),
        latency_ms: float = 0.0,
        request_id: str | None = None,
    ) -> None:
        if not self._settings.guardrails_audit_log_enabled or self._audit_logger is None:
            return

        chunk_data = [
            {
                "chunk_id": str(c.chunk_id),
                "document_id": str(c.document_id),
                "document_title": c.document_title,
                "chunk_index": c.chunk_index,
                "score": float(c.score),
            }
            for c in retrieved_chunks
        ]
        citation_data = [
            {
                "document_id": str(c.document_id),
                "document_title": c.document_title,
                "chunk_id": str(c.chunk_id),
                "chunk_index": c.chunk_index,
                "snippet": c.snippet,
            }
            for c in citations
        ]

        record = AuditRecord(
            tenant_id=tenant_id,
            question=processed_query,
            raw_question_hash=hash_query(raw_query),
            clearance=clearance,
            answer=answer,
            guardrail_action=action.value,
            retrieved_chunk_ids=chunk_data,
            citations=citation_data,
            guardrail_details=details,
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )
        await self._audit_logger.record(record)
