"""Unit tests for BM25 scoring and the PostgreSQL-backed keyword retriever."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from rag_engine.core.enums import AccessLevel
from rag_engine.core.models import Chunk
from rag_engine.retrieval.keyword import (
    BM25_B,
    BM25_K1,
    KeywordRetriever,
    bm25_score,
    selection_terms,
    tokenize,
)


def _chunk(*, content: str, char_count: int, index: int = 0) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        tenant_id=uuid4(),
        chunk_index=index,
        content=content,
        char_count=char_count,
        access_level=AccessLevel.INTERNAL,
    )


class ScriptedResult:
    def __init__(self, rows: list[Any], one: tuple[Any, ...] | None = None) -> None:
        self._rows = rows
        self._one = one

    def all(self) -> list[Any]:
        return self._rows

    def one(self) -> tuple[Any, ...]:
        assert self._one is not None, "statement did not expect a single row"
        return self._one


class ScriptedSession:
    """Returns canned results in call order and records the statements."""

    def __init__(self, results: list[ScriptedResult]) -> None:
        self._results = list(results)
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> ScriptedResult:
        self.statements.append(statement)
        return self._results.pop(0)


class ScriptedDatabase:
    def __init__(self, results: list[ScriptedResult]) -> None:
        self.session_instance = ScriptedSession(results)

    @asynccontextmanager
    async def session(self) -> Any:
        yield self.session_instance

    async def ping(self) -> float:
        return 1.0

    async def dispose(self) -> None:
        pass


def test_tokenize_matches_the_simple_text_search_lexemes() -> None:
    assert tokenize("Roll-back the K8s deployment_v2!") == [
        "roll",
        "back",
        "the",
        "k8s",
        "deployment_v2",
    ]
    assert tokenize("   ") == []


def test_bm25_defaults_are_the_classic_parameters() -> None:
    assert (BM25_K1, BM25_B) == (1.5, 0.75)


def test_bm25_ignores_absent_terms_and_penalises_long_chunks() -> None:
    document_frequencies = {"deploy": 2, "rollback": 1}

    short = bm25_score(
        term_frequencies={"deploy": 3},
        length=20,
        average_length=100.0,
        document_frequencies=document_frequencies,
        document_count=10,
    )
    long = bm25_score(
        term_frequencies={"deploy": 3},
        length=200,
        average_length=100.0,
        document_frequencies=document_frequencies,
        document_count=10,
    )
    absent = bm25_score(
        term_frequencies={"deploy": 1},
        length=20,
        average_length=100.0,
        document_frequencies={"rollback": 1},
        document_count=10,
    )

    assert short == pytest.approx(3.0867, abs=1e-4)
    assert long < short
    assert absent == 0.0


def test_bm25_ignores_terms_the_database_reports_as_absent() -> None:
    """A tokenisation mismatch must not fabricate a large IDF."""
    score = bm25_score(
        term_frequencies={"mismatch": 4},
        length=50,
        average_length=50.0,
        document_frequencies={"mismatch": 0},
        document_count=10,
    )

    assert score == 0.0


def test_bm25_handles_an_empty_corpus() -> None:
    assert (
        bm25_score(
            term_frequencies={"deploy": 1},
            length=10,
            average_length=0.0,
            document_frequencies={"deploy": 1},
            document_count=0,
        )
        == 0.0
    )


async def test_keyword_retriever_skips_queries_without_tokens() -> None:
    database = ScriptedDatabase([])
    retriever = KeywordRetriever(database=database)

    assert await retriever.retrieve(tenant_id=uuid4(), query="  !!! ??? ") == []
    assert database.session_instance.statements == []


async def test_keyword_retriever_returns_nothing_for_an_empty_tenant() -> None:
    database = ScriptedDatabase([ScriptedResult([], one=(0, None))])
    retriever = KeywordRetriever(database=database)

    assert await retriever.retrieve(tenant_id=uuid4(), query="deployment") == []
    # No document frequencies are needed when the visible corpus is empty.
    assert len(database.session_instance.statements) == 1


async def test_keyword_retriever_returns_nothing_when_no_candidates_match() -> None:
    database = ScriptedDatabase(
        [
            ScriptedResult([], one=(3, 100.0)),
            ScriptedResult([("deployment", 0)]),
            ScriptedResult([]),
        ]
    )
    retriever = KeywordRetriever(database=database)

    assert await retriever.retrieve(tenant_id=uuid4(), query="deployment") == []
    assert len(database.session_instance.statements) == 3


async def test_keyword_retriever_ranks_by_bm25_and_drops_zero_scores() -> None:
    deploy_only = _chunk(content="deploy deploy deploy", char_count=20, index=0)
    both_terms = _chunk(content="deploy rollback", char_count=15, index=1)
    unrelated = _chunk(content="receipts mileage", char_count=18, index=2)

    database = ScriptedDatabase(
        [
            ScriptedResult([], one=(10, 100.0)),
            ScriptedResult([("deploy", 2), ("rollback", 1)]),
            ScriptedResult(
                [
                    (deploy_only, "Runbook"),
                    (both_terms, "Runbook"),
                    (unrelated, "Travel"),
                ]
            ),
        ]
    )
    retriever = KeywordRetriever(database=database)

    results = await retriever.retrieve(tenant_id=uuid4(), query="deploy rollback", top_k=5)

    # "deploy rollback" carries both IDF terms and outranks the chunk that only
    # repeats one of them, despite being shorter.
    assert [item.chunk_index for item in results] == [1, 0]
    assert results[0].score > results[1].score > 0.0
    assert results[0].document_title == "Runbook"


async def test_keyword_retriever_breaks_ties_deterministically() -> None:
    zulu = _chunk(content="deploy", char_count=10, index=0)
    alpha = _chunk(content="deploy", char_count=10, index=1)

    def database() -> ScriptedDatabase:
        return ScriptedDatabase(
            [
                ScriptedResult([], one=(4, 10.0)),
                ScriptedResult([("deploy", 2)]),
                ScriptedResult([(zulu, "Zulu Handbook"), (alpha, "Alpha Handbook")]),
            ]
        )

    first = await KeywordRetriever(database=database()).retrieve(
        tenant_id=uuid4(), query="deploy", top_k=2
    )
    second = await KeywordRetriever(database=database()).retrieve(
        tenant_id=uuid4(), query="deploy", top_k=2
    )

    assert [item.document_title for item in first] == ["Alpha Handbook", "Zulu Handbook"]
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


async def test_keyword_retriever_applies_tenant_and_clearance_filters() -> None:
    database = ScriptedDatabase(
        [
            ScriptedResult([], one=(0, None)),
        ]
    )
    retriever = KeywordRetriever(database=database, candidate_limit=17)
    tenant_id = uuid4()

    await retriever.retrieve(
        tenant_id=tenant_id,
        query="deployment",
        clearance=AccessLevel.INTERNAL,
    )

    statement = database.session_instance.statements[0]
    sql = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )
    assert "chunks.tenant_id" in sql
    assert "chunks.access_level IN" in sql
    params = statement.compile(dialect=postgresql.dialect()).params
    assert tenant_id in params.values()


async def test_keyword_retriever_bounds_the_candidate_pool() -> None:
    database = ScriptedDatabase(
        [
            ScriptedResult([], one=(5, 100.0)),
            ScriptedResult([("deployment", 5)]),
            ScriptedResult([]),
        ]
    )
    retriever = KeywordRetriever(database=database, candidate_limit=17)

    await retriever.retrieve(tenant_id=uuid4(), query="deployment")

    candidate_statement = database.session_instance.statements[2]
    sql = str(candidate_statement.compile(dialect=postgresql.dialect()))
    params = candidate_statement.compile(dialect=postgresql.dialect()).params
    assert "LIMIT" in sql
    assert "chunks.tenant_id" in sql
    assert "chunks.access_level IN" in sql
    assert 17 in params.values()


async def test_keyword_retriever_ignores_the_dense_similarity_floor() -> None:
    """BM25 scores are not cosine similarities, so ``min_score`` must not filter."""
    chunk = _chunk(content="deploy deploy", char_count=12)
    database = ScriptedDatabase(
        [
            ScriptedResult([], one=(3, 12.0)),
            ScriptedResult([("deploy", 1)]),
            ScriptedResult([(chunk, "Runbook")]),
        ]
    )
    retriever = KeywordRetriever(database=database)

    results = await retriever.retrieve(tenant_id=uuid4(), query="deploy", min_score=0.99)

    assert len(results) == 1


async def test_keyword_retriever_rejects_a_non_positive_candidate_limit() -> None:
    with pytest.raises(ValueError, match="candidate_limit"):
        KeywordRetriever(database=ScriptedDatabase([]), candidate_limit=0)


def test_selection_terms_drop_corpus_common_words() -> None:
    # 10 visible chunks: "the" is in all of them, "deploy" in three.
    kept = selection_terms(
        ["the", "deploy", "rollback"],
        {"the": 10, "deploy": 3, "rollback": 1},
        document_count=10,
    )

    assert kept == ["deploy", "rollback"]


def test_selection_terms_never_drop_a_term_seen_in_one_chunk() -> None:
    # Two chunks, threshold is max(2, 1) = 2, so df=1 survives.
    assert selection_terms(["deploy", "the"], {"deploy": 1, "the": 2}, document_count=2) == [
        "deploy"
    ]


def test_selection_terms_fall_back_to_the_rarest_term() -> None:
    kept = selection_terms(
        ["the", "and", "of"],
        {"the": 50, "and": 40, "of": 30},
        document_count=50,
    )

    assert kept == ["of"]


def test_selection_terms_handle_an_empty_corpus() -> None:
    assert selection_terms(["deploy"], {"deploy": 0}, document_count=0) == []
