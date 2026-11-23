"""End-to-end integration test for Phase 4: Retrieval and Answer Generation.

Acceptance criteria:
- Upload document, ask question about it, get correct answer with working citations.
- "Not found" behavior when retrieval is weak or question is unanswerable from context.
- Clearance and tenant access control are enforced during query retrieval.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from httpx2 import AsyncClient

from rag_engine.generation.prompts import NOT_FOUND_RESPONSE
from rag_engine.ingestion.processor import IngestionProcessor

pytestmark = pytest.mark.integration

DOC_CONTENT = (
    "The Enterprise RAG Knowledge Engine uses Pinecone for dense vector storage. "
    "PostgreSQL is used as the primary relational database for metadata. "
    "Every chunk includes character spans, section headers, and page numbers. "
)


async def _upload(
    client: AsyncClient,
    headers: dict[str, str],
    *,
    filename: str = "spec.txt",
    data: bytes = DOC_CONTENT.encode(),
    access_level: str = "internal",
    title: str = "Architecture Specification",
) -> Any:
    return await client.post(
        "/documents",
        headers=headers,
        files={"file": (filename, data, "text/plain")},
        data={"access_level": access_level, "title": title},
    )


async def test_upload_document_and_query_cited_answer(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
) -> None:
    # 1. Upload document
    resp = await _upload(api_client, auth_headers)
    assert resp.status_code == 202
    doc_id = UUID(resp.json()["document_id"])

    # 2. Worker processes and indexes the document
    outcome = await processor.process_next()
    assert outcome.succeeded is True
    assert outcome.chunk_count > 0

    # 3. Query about content in the document
    query_resp = await api_client.post(
        "/query",
        headers=auth_headers,
        json={"question": "What is used for dense vector storage?"},
    )
    assert query_resp.status_code == 200, query_resp.text
    payload = query_resp.json()

    assert payload["question"] == "What is used for dense vector storage?"
    assert payload["weak_retrieval"] is False
    assert "Pinecone for dense vector storage" in payload["answer"]
    assert "[1]" in payload["answer"]

    assert len(payload["citations"]) >= 1
    citation = payload["citations"][0]
    assert citation["document_id"] == str(doc_id)
    assert citation["document_title"] == "Architecture Specification"
    assert citation["chunk_index"] == 0
    assert "Pinecone for dense vector storage" in citation["snippet"]


async def test_query_unanswerable_question_returns_not_found(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
) -> None:
    # Ingest document
    await _upload(api_client, auth_headers)
    assert (await processor.process_next()).succeeded is True

    # Ask unanswerable question
    query_resp = await api_client.post(
        "/query",
        headers=auth_headers,
        json={"question": "What is the capital of Mars and Jupiter?"},
    )
    assert query_resp.status_code == 200
    payload = query_resp.json()

    assert payload["answer"] == NOT_FOUND_RESPONSE
    assert payload["citations"] == []
    assert payload["weak_retrieval"] is True


async def test_query_respects_access_clearance(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
) -> None:
    # Upload a confidential document
    confidential_data = (
        "Project Orion is a secret initiative launched in 2026. "
        "The project lead is Dr. Elena Vance. "
    )
    upload_resp = await _upload(
        api_client,
        auth_headers,
        filename="orion.txt",
        data=confidential_data.encode(),
        access_level="confidential",
        title="Project Orion",
    )
    assert upload_resp.status_code == 202
    assert (await processor.process_next()).succeeded is True

    # Query with clearance=internal (cannot view confidential)
    internal_query = await api_client.post(
        "/query",
        headers=auth_headers,
        json={
            "question": "Who is the project lead for Project Orion?",
            "clearance": "internal",
        },
    )
    assert internal_query.status_code == 200
    assert internal_query.json()["weak_retrieval"] is True
    assert internal_query.json()["answer"] == NOT_FOUND_RESPONSE

    # Query with clearance=confidential (can view confidential)
    confidential_query = await api_client.post(
        "/query",
        headers=auth_headers,
        json={
            "question": "Who is the project lead for Project Orion?",
            "clearance": "confidential",
        },
    )
    assert confidential_query.status_code == 200
    conf_payload = confidential_query.json()
    assert conf_payload["weak_retrieval"] is False
    assert "Elena Vance" in conf_payload["answer"]
    assert len(conf_payload["citations"]) >= 1
    assert conf_payload["citations"][0]["document_title"] == "Project Orion"
