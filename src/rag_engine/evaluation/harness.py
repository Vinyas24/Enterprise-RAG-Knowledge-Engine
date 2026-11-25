"""Run a dataset through the retrieval pipeline in an isolated tenant.

The corpus is indexed directly instead of being uploaded through the job
queue. That is deliberate: an evaluation must be repeatable even when a worker
is running against the same database, and ingestion correctness is already
covered by the ingestion tests. Chunking, embedding, ids and vector metadata
still come from the production modules, so retrieval is measured on the same
index shape production uses.

Each run gets a fresh tenant and removes its rows and vectors afterwards, even
when a question or indexing step fails.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from rag_engine.core.config import RetrievalMode, Settings
from rag_engine.core.db import DatabaseProtocol
from rag_engine.core.enums import DocumentStatus
from rag_engine.core.ids import chunk_vector_id
from rag_engine.core.models import Chunk, Document, Tenant
from rag_engine.embedding import Embedder
from rag_engine.ingestion import repository
from rag_engine.ingestion.chunking import build_chunker
from rag_engine.ingestion.hashing import hash_content
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.vectorize import build_vector_records
from rag_engine.retrieval import build_retriever
from rag_engine.vectorstore import VectorStore

from .dataset import EvalDataset, EvalQuestion, dataset_fingerprint
from .metrics import (
    QuestionResult,
    RetrievedItem,
    aggregate_metrics,
    compare_metrics,
    evaluate_question,
)
from .report import DatasetSummary, EvaluationConfig, EvaluationReport

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """The dataset could not be indexed or evaluated."""


async def evaluate_dataset(
    dataset: EvalDataset,
    *,
    settings: Settings,
    database: DatabaseProtocol,
    vector_store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    min_score: float | None = None,
    baseline: EvaluationReport | None = None,
) -> EvaluationReport:
    """Index ``dataset`` in an isolated tenant, retrieve, and score it."""
    resolved_k = top_k if top_k is not None else settings.retrieval_top_k
    if resolved_k < 1:
        raise EvaluationError("top_k must be at least 1")
    resolved_min_score = settings.retrieval_min_score if min_score is None else min_score

    tenant = await _create_eval_tenant(database, dataset.name)
    namespace = str(tenant.id)
    cleanup_warnings: list[str] = []

    try:
        document_ids, chunk_count = await _index_corpus(
            dataset,
            tenant=tenant,
            database=database,
            vector_store=vector_store,
            embedder=embedder,
            settings=settings,
        )
        key_by_id = {document_id: key for key, document_id in document_ids.items()}
        results = await _run_questions(
            dataset.questions,
            tenant=tenant,
            database=database,
            vector_store=vector_store,
            embedder=embedder,
            settings=settings,
            key_by_id=key_by_id,
            top_k=resolved_k,
            min_score=resolved_min_score,
        )
    finally:
        cleanup_warnings = await _cleanup(
            database, vector_store, tenant=tenant, namespace=namespace
        )

    metrics = aggregate_metrics(results, k=resolved_k)
    fingerprint = dataset_fingerprint(dataset)
    hybrid = settings.retrieval_mode is RetrievalMode.HYBRID
    config = EvaluationConfig(
        retriever=settings.retrieval_mode.value,
        embedding_provider=settings.embedding_provider,
        embedding_dimensions=settings.embedding_dimensions,
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        vector_store=settings.vector_store,
        top_k=resolved_k,
        min_score=resolved_min_score,
        keyword_candidates=settings.retrieval_keyword_candidates if hybrid else None,
        hybrid_dense_weight=settings.hybrid_dense_weight if hybrid else None,
        hybrid_keyword_weight=settings.hybrid_keyword_weight if hybrid else None,
        hybrid_rrf_k=settings.hybrid_rrf_k if hybrid else None,
    )
    warnings = list(cleanup_warnings)
    if not 30 <= len(dataset.questions) <= 50:
        warnings.append(
            f"dataset has {len(dataset.questions)} questions; the Phase 5 target is 30-50"
        )
    if baseline is not None:
        warnings.extend(_baseline_warnings(baseline, fingerprint, config))

    return EvaluationReport(
        generated_at=datetime.now(UTC),
        dataset=DatasetSummary(
            name=dataset.name,
            version=dataset.version,
            fingerprint=fingerprint,
            document_count=len(dataset.documents),
            chunk_count=chunk_count,
            question_count=len(dataset.questions),
        ),
        config=config,
        metrics=metrics,
        per_question=results,
        comparison=(compare_metrics(metrics, baseline.metrics) if baseline is not None else []),
        warnings=warnings,
    )


async def _create_eval_tenant(database: DatabaseProtocol, name: str) -> Tenant:
    tenant = Tenant(
        slug=f"eval-{uuid4().hex[:12]}",
        name=f"Evaluation: {name}"[:255],
    )
    async with database.session() as session:
        session.add(tenant)
        await session.commit()
    return tenant


async def _index_corpus(
    dataset: EvalDataset,
    *,
    tenant: Tenant,
    database: DatabaseProtocol,
    vector_store: VectorStore,
    embedder: Embedder,
    settings: Settings,
) -> tuple[dict[str, UUID], int]:
    """Chunk, embed and persist the corpus, returning ids and chunk count."""
    parsers = ParserRegistry.default()
    chunker = build_chunker(settings)
    namespace = str(tenant.id)
    document_ids: dict[str, UUID] = {}
    chunk_count = 0

    for document_spec in dataset.documents:
        data = document_spec.content.encode("utf-8")
        parsed = parsers.parser_for(document_spec.filename).parse(data)
        drafts = chunker.chunk(parsed)
        if not drafts:
            raise EvaluationError(f"document {document_spec.key!r} produced no chunks")
        vectors = await embedder.embed([draft.content for draft in drafts])

        document = Document(
            tenant_id=tenant.id,
            title=document_spec.title,
            filename=document_spec.filename,
            content_type="text/markdown",
            size_bytes=len(data),
            content_hash=hash_content(data),
            storage_key=f"eval/{document_spec.key}",
            access_level=document_spec.access_level,
            status=DocumentStatus.PENDING,
        )
        async with database.session() as session:
            session.add(document)
            try:
                await session.flush()
            except IntegrityError as exc:
                raise EvaluationError(
                    f"document {document_spec.key!r} has the same content as "
                    "an earlier dataset document; make each document unique"
                ) from exc

            records = build_vector_records(
                document_id=document.id,
                tenant_id=tenant.id,
                drafts=drafts,
                vectors=vectors,
                access_level=document_spec.access_level,
                dimensions=embedder.dimensions,
            )
            # Vectors first, matching the worker's fail-closed ordering.
            await vector_store.upsert(namespace, records)
            await repository.replace_chunks(
                session,
                document_id=document.id,
                tenant_id=tenant.id,
                drafts=drafts,
                access_level=document_spec.access_level,
            )
            document.status = DocumentStatus.INDEXED
            await session.commit()

        document_ids[document_spec.key] = document.id
        chunk_count += len(drafts)

    return document_ids, chunk_count


async def _run_questions(
    questions: list[EvalQuestion],
    *,
    tenant: Tenant,
    database: DatabaseProtocol,
    vector_store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    key_by_id: dict[UUID, str],
    top_k: int,
    min_score: float,
) -> list[QuestionResult]:
    retriever = build_retriever(
        settings, embedder=embedder, vector_store=vector_store, database=database
    )
    # One untimed call warms the embedding path and the connection pool so the
    # first question is not charged for process-level setup.
    await retriever.retrieve(
        tenant_id=tenant.id,
        query=questions[0].question,
        top_k=top_k,
        clearance=questions[0].clearance,
        min_score=min_score,
    )

    results: list[QuestionResult] = []
    for question in questions:
        started = time.perf_counter()
        chunks = await retriever.retrieve(
            tenant_id=tenant.id,
            query=question.question,
            top_k=top_k,
            clearance=question.clearance,
            min_score=min_score,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        retrieved: list[RetrievedItem] = []
        for chunk in chunks:
            key = key_by_id.get(chunk.document_id)
            if key is None:
                raise EvaluationError(
                    f"retrieval returned a chunk from an unknown document {chunk.document_id}"
                )
            retrieved.append(
                RetrievedItem(
                    document_key=key,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                )
            )
        results.append(evaluate_question(question, retrieved, k=top_k, latency_ms=elapsed_ms))
    return results


async def _cleanup(
    database: DatabaseProtocol,
    vector_store: VectorStore,
    *,
    tenant: Tenant,
    namespace: str,
) -> list[str]:
    warnings: list[str] = []
    known_ids: list[str] = []
    try:
        async with database.session() as session:
            rows = (
                await session.execute(
                    select(Chunk.document_id, Chunk.chunk_index).where(Chunk.tenant_id == tenant.id)
                )
            ).all()
        known_ids = [chunk_vector_id(document_id, index) for document_id, index in rows]
    except Exception:
        logger.warning("evaluation.chunk_lookup_failed", exc_info=True)

    try:
        if known_ids:
            await vector_store.delete(namespace, known_ids)
        remaining = await vector_store.list_ids(namespace)
        if remaining:
            await vector_store.delete(namespace, remaining)
    except Exception as exc:
        logger.warning("evaluation.vector_cleanup_failed", exc_info=True)
        warnings.append(f"vector cleanup failed for evaluation namespace {namespace}: {exc}")

    try:
        async with database.session() as session:
            await session.execute(delete(Tenant).where(Tenant.id == tenant.id))
            await session.commit()
    except Exception as exc:
        logger.warning("evaluation.tenant_cleanup_failed", exc_info=True)
        warnings.append(f"tenant cleanup failed for {tenant.slug}: {exc}")

    return warnings


def _baseline_warnings(
    baseline: EvaluationReport,
    fingerprint: str,
    config: EvaluationConfig,
) -> list[str]:
    warnings: list[str] = []
    if baseline.dataset.fingerprint != fingerprint:
        warnings.append(
            "baseline was recorded on a different dataset fingerprint; "
            "deltas include dataset changes, not only retrieval changes"
        )
    changed = [
        name
        for name in type(config).model_fields
        if name != "retriever" and getattr(config, name) != getattr(baseline.config, name)
    ]
    if changed:
        warnings.append("baseline configuration differs: " + ", ".join(sorted(changed)))
    if baseline.config.retriever != config.retriever:
        warnings.append(
            f"baseline used retriever {baseline.config.retriever!r} and this run used "
            f"{config.retriever!r}; the delta is the effect of that change, not a "
            "like-for-like comparison"
        )
    return warnings
