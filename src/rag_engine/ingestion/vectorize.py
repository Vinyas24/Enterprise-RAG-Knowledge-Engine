"""Build vector records from chunk drafts and embeddings.

Shared by the ingestion worker and the evaluation harness so both write the
same ids and metadata. A mismatch here would make evaluation measure a
different index than production, which is exactly the kind of silent drift the
harness exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from rag_engine.core.enums import AccessLevel
from rag_engine.core.ids import chunk_vector_id
from rag_engine.ingestion.chunking import ChunkDraft
from rag_engine.ingestion.errors import IngestionError
from rag_engine.vectorstore import VectorRecord, metadata_for_vector


def build_vector_records(
    *,
    document_id: UUID,
    tenant_id: UUID,
    drafts: Sequence[ChunkDraft],
    vectors: Sequence[Sequence[float]],
    access_level: AccessLevel,
    dimensions: int,
) -> list[VectorRecord]:
    """Pair each draft with its embedding and the metadata retrieval needs."""
    if len(vectors) != len(drafts):
        raise IngestionError(f"embedder returned {len(vectors)} vectors for {len(drafts)} chunks")

    records: list[VectorRecord] = []
    for index, (draft, vector) in enumerate(zip(drafts, vectors, strict=True)):
        if len(vector) != dimensions:
            raise IngestionError(
                f"embedding {index} has {len(vector)} dimensions, expected {dimensions}"
            )
        records.append(
            VectorRecord(
                id=chunk_vector_id(document_id, index),
                values=list(vector),
                metadata=metadata_for_vector(
                    {
                        "document_id": str(document_id),
                        "tenant_id": str(tenant_id),
                        "chunk_index": index,
                        "access_level": str(access_level),
                        "page_start": draft.page_start,
                        "page_end": draft.page_end,
                        "section_path": draft.section_path,
                        "start_char": draft.start_char,
                        "end_char": draft.end_char,
                    }
                ),
            )
        )
    return records
