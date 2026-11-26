"""BM25 keyword retrieval backed by PostgreSQL full-text search.

**Where the index lives.** PostgreSQL, not the application and not the vector
store. Three reasons decided it:

* Chunks already live there. A keyword index built in-process would need a
  second write path at ingestion time and would drift from the rows retrieval
  hydrates; an index the database maintains cannot.
* The same transaction that writes a chunk writes its lexemes, so a rolled-back
  ingestion leaves no keyword entries behind.
* Tenant and clearance predicates are ordinary ``WHERE`` clauses on the same
  table. Dense retrieval has to trust vector metadata and then re-check it in
  SQL; the keyword path filters exactly once, in the query that selects
  candidates.

Pinecone sparse vectors were the other option. They would keep one backend,
but they push tenant/clearance filtering into metadata filters, cannot compute
corpus-wide document frequencies cheaply, and tie the keyword path to a
provider that local development only emulates.

**How the score is computed.** PostgreSQL does candidate selection (``@@``
with a GIN index) and BM25 is evaluated in Python over those candidates. This
is a deliberate split:

* ``ts_rank`` is not BM25 — it has no document-length or corpus-frequency
  terms. Relabelling it as BM25 would be dishonest.
* The two halves of BM25 need different data: term frequency per chunk (cheap
  in Python from the content already fetched) and document frequency plus
  corpus size (cheap in SQL with the index).

The candidate pool is bounded, so BM25 is exact for the chunks PostgreSQL
selects and a heuristic only about *which* chunks get scored. ``char_count``
is the document-length proxy: reusing a column ingestion already writes beats
adding a token-count column that would have to be backfilled.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.core.db import DatabaseProtocol
from rag_engine.core.enums import AccessLevel, visible_access_levels
from rag_engine.core.models import Chunk, Document

from .base import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

# Tuned to the classic defaults. ``k1`` controls how quickly term frequency
# saturates; ``b`` controls how strongly longer chunks are penalised.
BM25_K1 = 1.5
BM25_B = 0.75

# A term that appears in at least half the visible corpus (and in at least two
# chunks) is dropped from *candidate selection*: with OR semantics it would
# match nearly every chunk and dilute the bounded candidate pool. BM25 already
# gives such a term an IDF near zero, so scoring it changes nothing.
_MAX_DOCUMENT_FREQUENCY_FRACTION = 0.5

# PostgreSQL's ``simple`` configuration treats letters, digits and underscore
# as word characters. Matching that here keeps a Python token and a lexeme the
# same string, which is what makes per-chunk term frequency comparable with
# the document frequencies the database reports.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# ``to_tsvector``/``to_tsquery`` are only immutable (and therefore usable in a
# generated column and an index) when the text-search configuration is passed
# explicitly, so the config is inlined rather than bound.
_SIMPLE_CONFIG: sa.ColumnElement[str] = sa.literal_column("'simple'::regconfig")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into the lexemes the ``simple`` configuration produces."""
    return _TOKEN_RE.findall(text.lower())


def bm25_score(
    *,
    term_frequencies: Mapping[str, int],
    length: int,
    average_length: float,
    document_frequencies: Mapping[str, int],
    document_count: int,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """BM25 score for one chunk.

    Only terms with a positive document frequency contribute. A term the
    database reports as absent cannot have come from a lexeme the query
    matched, so scoring it would invent a large IDF out of a tokenisation
    mismatch.
    """
    if document_count <= 0:
        return 0.0
    normalised_length = length / average_length if average_length > 0 else 0.0
    score = 0.0
    for term, df in document_frequencies.items():
        if df <= 0:
            continue
        tf = term_frequencies.get(term, 0)
        if tf <= 0:
            continue
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        denominator = tf + k1 * (1.0 - b + b * normalised_length)
        score += idf * (tf * (k1 + 1.0)) / denominator
    return score


def selection_terms(
    terms: Sequence[str],
    document_frequencies: Mapping[str, int],
    document_count: int,
) -> list[str]:
    """Drop corpus-common terms from candidate selection.

    Falls back to the rarest term when every term is common, so a query made
    only of very common words still produces a (small) candidate set instead
    of silently returning nothing.
    """
    if document_count <= 0:
        return []
    threshold = max(2, math.ceil(document_count * _MAX_DOCUMENT_FREQUENCY_FRACTION))
    kept = [term for term in terms if document_frequencies.get(term, 0) < threshold]
    if kept:
        return kept
    return [min(terms, key=lambda term: document_frequencies.get(term, 0))]


class KeywordRetriever(Retriever):
    """Lexical retrieval: PostgreSQL candidate selection, BM25 ranking."""

    def __init__(
        self,
        *,
        database: DatabaseProtocol,
        candidate_limit: int = 200,
        timeout_seconds: float | None = None,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be at least 1")
        self._database = database
        self._candidate_limit = candidate_limit
        self._timeout_seconds = timeout_seconds

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks whose lexemes match the query, ranked by BM25.

        ``min_score`` is accepted for interface compatibility but ignored: it is
        a cosine-similarity floor and BM25 scores are on an unrelated scale.
        Thresholding them with it would silently discard every result.
        """
        if self._timeout_seconds is not None:
            return await asyncio.wait_for(
                self._retrieve_internal(
                    tenant_id=tenant_id,
                    query=query,
                    top_k=top_k,
                    clearance=clearance,
                ),
                timeout=self._timeout_seconds,
            )
        return await self._retrieve_internal(
            tenant_id=tenant_id,
            query=query,
            top_k=top_k,
            clearance=clearance,
        )

    async def _retrieve_internal(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int,
        clearance: AccessLevel,
    ) -> list[RetrievedChunk]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []

        allowed_levels = visible_access_levels(clearance)
        async with self._database.session() as session:
            document_count, average_length = await self._corpus_shape(
                session, tenant_id=tenant_id, levels=allowed_levels
            )
            if document_count == 0:
                return []
            document_frequencies = await self._document_frequencies(
                session, tenant_id=tenant_id, levels=allowed_levels, terms=terms
            )
            candidates = await self._candidates(
                session,
                tenant_id=tenant_id,
                levels=allowed_levels,
                terms=selection_terms(terms, document_frequencies, document_count),
            )

        retrieved: list[RetrievedChunk] = []
        for chunk, document_title in candidates:
            score = bm25_score(
                term_frequencies=Counter(tokenize(chunk.content)),
                length=chunk.char_count,
                average_length=average_length,
                document_frequencies=document_frequencies,
                document_count=document_count,
            )
            if score <= 0.0:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=document_title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=score,
                    access_level=chunk.access_level,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                )
            )

        retrieved.sort(
            key=lambda item: (
                -item.score,
                item.document_title,
                item.chunk_index,
                item.content,
            )
        )
        return retrieved[:top_k]

    async def _corpus_shape(
        self,
        session: AsyncSession,
        *,
        tenant_id: UUID,
        levels: Sequence[AccessLevel],
    ) -> tuple[int, float]:
        """Chunk count and mean length inside the caller's visible scope."""
        statement = select(func.count(), func.avg(Chunk.char_count)).where(
            Chunk.tenant_id == tenant_id,
            Chunk.access_level.in_(levels),
        )
        document_count, average_length = (await session.execute(statement)).one()
        resolved_average = float(average_length) if average_length is not None else 0.0
        return int(document_count or 0), resolved_average

    async def _document_frequencies(
        self,
        session: AsyncSession,
        *,
        tenant_id: UUID,
        levels: Sequence[AccessLevel],
        terms: Sequence[str],
    ) -> dict[str, int]:
        """How many visible chunks contain each query term.

        Measured inside the caller's scope: using the whole table would let
        restricted chunks shift the score of chunks a caller can read, which
        leaks their existence.
        """
        # One round trip for every query term. ``unnest`` turns the bound array
        # into rows, and the correlated count reuses the GIN index per term.
        statement = sa.text(
            """
            SELECT term, (
                SELECT count(*)
                FROM chunks AS c
                WHERE c.tenant_id = :tenant_id
                  AND c.access_level IN :levels
                  AND c.content_tsv @@ to_tsquery('simple'::regconfig, term)
            ) AS df
            FROM unnest(:terms) AS term
            """
        ).bindparams(
            sa.bindparam("tenant_id", value=tenant_id, type_=Chunk.tenant_id.type),
            sa.bindparam("levels", value=[level.value for level in levels], expanding=True),
            sa.bindparam("terms", value=list(terms), type_=postgresql.ARRAY(sa.Text())),
        )
        return {str(term): int(df) for term, df in (await session.execute(statement)).all()}

    async def _candidates(
        self,
        session: AsyncSession,
        *,
        tenant_id: UUID,
        levels: Sequence[AccessLevel],
        terms: Sequence[str],
    ) -> list[tuple[Chunk, str]]:
        """Select at most ``candidate_limit`` chunks matching ``terms``.

        Ordering by ``ts_rank`` is a cheap heuristic that only decides which
        candidates survive the cut. It is deliberately *not* the final score:
        ``ts_rank`` ignores corpus statistics and document length, and
        ``ts_rank_cd`` costs roughly ten times as much to compute.
        """
        if not terms:
            return []
        tsquery = func.to_tsquery(_SIMPLE_CONFIG, " | ".join(terms))
        rank = func.ts_rank(Chunk.content_tsv, tsquery).label("fts_rank")
        statement = (
            select(Chunk, Document.title)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.access_level.in_(levels),
                Chunk.content_tsv.op("@@")(tsquery),
            )
            .order_by(rank.desc(), Chunk.id)
            .limit(self._candidate_limit)
        )
        rows = (await session.execute(statement)).all()
        return [(chunk, title) for chunk, title in rows]
