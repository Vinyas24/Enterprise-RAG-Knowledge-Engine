"""Question-answering query endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from rag_engine.api.deps import get_query_engine, get_tenant
from rag_engine.api.schemas import CitationRead, QueryRequest, QueryResponse
from rag_engine.core.models import Tenant
from rag_engine.generation.service import QueryEngine

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the knowledge engine for a cited answer",
    description=(
        "Embeds the question, retrieves top matching chunks under the caller's "
        "tenant and access clearance, assembles a strict factual prompt, and returns "
        "the synthesized answer alongside source citations."
    ),
)
async def query_knowledge_base(
    payload: QueryRequest,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    query_engine: Annotated[QueryEngine, Depends(get_query_engine)],
) -> QueryResponse:
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

    return QueryResponse(
        question=payload.question,
        answer=result.answer,
        citations=citations,
        weak_retrieval=result.weak_retrieval,
    )
