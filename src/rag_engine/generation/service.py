"""Query and answer generation service."""

from __future__ import annotations

import logging
import time
from uuid import UUID

from rag_engine.api.errors import ApiError
from rag_engine.core.config import Settings
from rag_engine.core.context import get_request_id
from rag_engine.core.enums import AccessLevel
from rag_engine.guardrails.base import GuardrailAction
from rag_engine.guardrails.service import INJECTION_REFUSAL_RESPONSE, GuardrailsService
from rag_engine.observability import (
    PipelineTracer,
    get_current_tracer,
    get_metrics_registry,
    reset_current_tracer,
    set_current_tracer,
    trace_stage,
)
from rag_engine.retrieval.base import Retriever

from .base import GenerationResult, LLMProvider
from .citations import extract_citations
from .prompts import NOT_FOUND_RESPONSE, assemble_prompt

logger = logging.getLogger(__name__)


class QueryEngine:
    """Coordinates retrieval, prompt assembly, LLM generation, guardrails, and citation mapping."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        settings: Settings,
        guardrails: GuardrailsService | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings
        self._guardrails = guardrails

    async def answer(
        self,
        *,
        tenant_id: UUID,
        question: str,
        top_k: int | None = None,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        request_id: str | None = None,
    ) -> GenerationResult:
        start_time = time.perf_counter()
        req_id = request_id or get_request_id()
        resolved_top_k = top_k or self._settings.retrieval_top_k
        clean_question = question.strip()
        processed_question = clean_question

        tracer = get_current_tracer()
        created_tracer = False
        token = None
        if tracer is None:
            tracer = PipelineTracer()
            token = set_current_tracer(tracer)
            created_tracer = True

        try:
            # 1. Input Guardrails (Rate limit, length check, injection detection, PII masking)
            if self._guardrails is not None:
                t_input = time.perf_counter()
                with trace_stage("guardrails_input"):
                    processed_question, input_res = await self._guardrails.check_input(
                        tenant_id=tenant_id,
                        query=clean_question,
                        clearance=clearance,
                        request_id=req_id,
                    )
                dur_input = time.perf_counter() - t_input
                get_metrics_registry().record_timing(
                    "rag_pipeline_stage_duration_seconds",
                    dur_input,
                    labels={"stage": "guardrails_input"},
                )
                if not input_res.passed:
                    if input_res.action == GuardrailAction.RATE_LIMITED:
                        raise ApiError(
                            status_code=429,
                            code="rate_limited",
                            message=input_res.reason or "Rate limit exceeded.",
                            details=input_res.details,
                        )
                    if input_res.action == GuardrailAction.INPUT_LENGTH_EXCEEDED:
                        raise ApiError(
                            status_code=400,
                            code="query_too_long",
                            message=input_res.reason or "Query length exceeded.",
                            details=input_res.details,
                        )
                    if input_res.action == GuardrailAction.REFUSED_INJECTION:
                        total_dur = (time.perf_counter() - start_time) * 1000.0
                        tracer.record_stage("total", total_dur)
                        return GenerationResult(
                            answer=INJECTION_REFUSAL_RESPONSE,
                            citations=[],
                            weak_retrieval=True,
                            question=processed_question,
                            timings=tracer.to_dict(),
                        )

            # 2. Retrieve candidate chunks under strict tenant and clearance filters
            t_ret = time.perf_counter()
            with trace_stage("retrieve"):
                passages = await self._retriever.retrieve(
                    tenant_id=tenant_id,
                    query=processed_question,
                    top_k=resolved_top_k,
                    clearance=clearance,
                    min_score=self._settings.retrieval_min_score,
                )
            dur_ret = time.perf_counter() - t_ret
            get_metrics_registry().record_timing(
                "rag_pipeline_stage_duration_seconds", dur_ret, labels={"stage": "retrieve"}
            )

            # 3. Check for weak retrieval
            if not passages:
                logger.info("generation.weak_retrieval", extra={"tenant_id": str(tenant_id)})
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                if self._guardrails is not None:
                    await self._guardrails.check_output(
                        tenant_id=tenant_id,
                        query=processed_question,
                        answer=NOT_FOUND_RESPONSE,
                        passages=[],
                        citations=[],
                        clearance=clearance,
                        request_id=req_id,
                        latency_ms=elapsed_ms,
                    )
                tracer.record_stage("total", elapsed_ms)
                return GenerationResult(
                    answer=NOT_FOUND_RESPONSE,
                    citations=[],
                    weak_retrieval=True,
                    question=processed_question,
                    timings=tracer.to_dict(),
                )

            # 4. Indirect document injection defense (sanitize passage boundary delimiters)
            if self._guardrails is not None:
                passages = self._guardrails.sanitize_passages(passages)

            # 5. Assemble strict contextual prompt
            system_prompt, user_prompt = assemble_prompt(processed_question, passages)

            # 6. Generate answer
            t_gen = time.perf_counter()
            with trace_stage("generate"):
                raw_answer = await self._llm.generate(user_prompt, system_prompt=system_prompt)
            dur_gen = time.perf_counter() - t_gen
            get_metrics_registry().record_timing(
                "rag_pipeline_stage_duration_seconds", dur_gen, labels={"stage": "generate"}
            )

            # 7. Check if LLM determined the context does not answer the question
            if raw_answer.strip() == NOT_FOUND_RESPONSE or raw_answer.startswith("I cannot answer"):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                if self._guardrails is not None:
                    await self._guardrails.check_output(
                        tenant_id=tenant_id,
                        query=processed_question,
                        answer=NOT_FOUND_RESPONSE,
                        passages=passages,
                        citations=[],
                        clearance=clearance,
                        request_id=req_id,
                        latency_ms=elapsed_ms,
                    )
                tracer.record_stage("total", elapsed_ms)
                return GenerationResult(
                    answer=NOT_FOUND_RESPONSE,
                    citations=[],
                    weak_retrieval=True,
                    question=processed_question,
                    timings=tracer.to_dict(),
                )

            # 8. Extract citations from answer
            citations = extract_citations(raw_answer, passages)

            # 9. Output Guardrails (citations, groundedness, injection defense)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if self._guardrails is not None:
                t_out = time.perf_counter()
                with trace_stage("guardrails_output"):
                    (
                        final_answer,
                        valid_citations,
                        weak_retrieval,
                        _,
                    ) = await self._guardrails.check_output(
                        tenant_id=tenant_id,
                        query=processed_question,
                        answer=raw_answer,
                        passages=passages,
                        citations=citations,
                        clearance=clearance,
                        request_id=req_id,
                        latency_ms=elapsed_ms,
                    )
                dur_out = time.perf_counter() - t_out
                get_metrics_registry().record_timing(
                    "rag_pipeline_stage_duration_seconds",
                    dur_out,
                    labels={"stage": "guardrails_output"},
                )
                total_dur = (time.perf_counter() - start_time) * 1000.0
                tracer.record_stage("total", total_dur)
                return GenerationResult(
                    answer=final_answer,
                    citations=valid_citations,
                    weak_retrieval=weak_retrieval,
                    question=processed_question,
                    timings=tracer.to_dict(),
                )

            total_dur = (time.perf_counter() - start_time) * 1000.0
            tracer.record_stage("total", total_dur)
            return GenerationResult(
                answer=raw_answer,
                citations=citations,
                weak_retrieval=False,
                question=processed_question,
                timings=tracer.to_dict(),
            )
        finally:
            if created_tracer and token is not None:
                reset_current_tracer(token)
