"""Shared vector-record construction used by the worker and the harness."""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_engine.core.enums import AccessLevel
from rag_engine.core.ids import chunk_vector_id
from rag_engine.ingestion.chunking import ChunkDraft
from rag_engine.ingestion.errors import IngestionError
from rag_engine.ingestion.vectorize import build_vector_records


def _draft(content: str) -> ChunkDraft:
    return ChunkDraft(
        content=content,
        char_count=len(content),
        page_start=1,
        page_end=1,
        section_path="Policy",
        start_char=0,
        end_char=len(content),
    )


def test_records_use_deterministic_ids_and_retrieval_metadata() -> None:
    document_id = uuid4()
    tenant_id = uuid4()

    records = build_vector_records(
        document_id=document_id,
        tenant_id=tenant_id,
        drafts=[_draft("first"), _draft("second")],
        vectors=[[0.1] * 4, [0.2] * 4],
        access_level=AccessLevel.CONFIDENTIAL,
        dimensions=4,
    )

    assert [record.id for record in records] == [
        chunk_vector_id(document_id, 0),
        chunk_vector_id(document_id, 1),
    ]
    assert records[0].values == [0.1] * 4
    assert records[1].metadata["chunk_index"] == 1
    assert records[1].metadata["document_id"] == str(document_id)
    assert records[1].metadata["tenant_id"] == str(tenant_id)
    assert records[1].metadata["access_level"] == "confidential"
    assert records[1].metadata["section_path"] == "Policy"


def test_vector_count_mismatch_is_rejected() -> None:
    with pytest.raises(IngestionError, match="returned 1 vectors for 2 chunks"):
        build_vector_records(
            document_id=uuid4(),
            tenant_id=uuid4(),
            drafts=[_draft("first"), _draft("second")],
            vectors=[[0.1] * 4],
            access_level=AccessLevel.INTERNAL,
            dimensions=4,
        )


def test_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(IngestionError, match="dimensions, expected 4"):
        build_vector_records(
            document_id=uuid4(),
            tenant_id=uuid4(),
            drafts=[_draft("first")],
            vectors=[[0.1] * 3],
            access_level=AccessLevel.INTERNAL,
            dimensions=4,
        )
