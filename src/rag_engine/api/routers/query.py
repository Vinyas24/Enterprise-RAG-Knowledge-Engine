"""Question-answering query endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from rag_engine.api.auth import require_api_key
from rag_engine.api.deps import get_app_settings, get_query_engine, get_tenant
from rag_engine.api.errors import COMMON_ERROR_RESPONSES
from rag_engine.api.schemas import CitationRead, QueryRequest, QueryResponse, QueryTimings
from rag_engine.core.config import Settings
from rag_engine.core.models import Tenant
from rag_engine.generation.service import QueryEngine
from rag_engine.observability import (
    PipelineTracer,
    reset_current_tracer,
    set_current_tracer,
)

router = APIRouter(
    prefix="/query",
    tags=["query"],
    dependencies=[Depends(require_api_key)],
    responses=COMMON_ERROR_RESPONSES,
)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the knowledge engine for a cited answer",
    description=(
        "Embeds the question, retrieves top matching chunks under the caller's "
        "tenant and access clearance, assembles a strict factual prompt, and returns "
        "the synthesized answer alongside source citations and stage latency breakdown."
    ),
)
async def query_knowledge_base(
    payload: QueryRequest,
    response: Response,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    query_engine: Annotated[QueryEngine, Depends(get_query_engine)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> QueryResponse:
    tracer = PipelineTracer()
    token = set_current_tracer(tracer)
    try:
        result = await query_engine.answer(
            tenant_id=tenant.id,
            question=payload.question,
            top_k=payload.top_k,
            clearance=payload.clearance,
        )

        citations = [
            CitationRead(
                document_id=c.document_id,
                document_title=c.document_title,
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                snippet=c.snippet,
                page_start=c.page_start,
                page_end=c.page_end,
                section_path=c.section_path,
            )
            for c in result.citations
        ]

        timings: QueryTimings | None = None
        if result.timings:
            timings = QueryTimings(
                total_ms=result.timings.get("total", 0.0),
                retrieve_ms=result.timings.get("retrieve", 0.0),
                dense_ms=result.timings.get("dense"),
                keyword_ms=result.timings.get("keyword"),
                fuse_ms=result.timings.get("fuse"),
                rerank_ms=result.timings.get("rerank"),
                generate_ms=result.timings.get("generate"),
                guardrails_input_ms=result.timings.get("guardrails_input", 0.0),
                guardrails_output_ms=result.timings.get("guardrails_output", 0.0),
            )

        if settings.server_timing_enabled:
            header_val = tracer.to_server_timing_header()
            if header_val:
                response.headers["Server-Timing"] = header_val

        return QueryResponse(
            question=result.question or payload.question,
            answer=result.answer,
            citations=citations,
            weak_retrieval=result.weak_retrieval,
            timings=timings,
        )
    finally:
        reset_current_tracer(token)
