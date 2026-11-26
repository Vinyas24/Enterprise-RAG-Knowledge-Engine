"""End-to-end evaluation against real PostgreSQL and the in-memory store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from rag_engine.core.models import Tenant
from rag_engine.evaluation import harness
from rag_engine.evaluation.dataset import EvalDataset
from rag_engine.evaluation.harness import EvaluationError, evaluate_dataset
from rag_engine.evaluation.report import render_markdown
from rag_engine.ranking import RerankTransientError, RerankUnavailableError

pytestmark = pytest.mark.integration


def _dataset() -> EvalDataset:
    """Two documents with deliberately disjoint vocabulary.

    The deterministic embedder is a bag of words, so a question using only one
    document's vocabulary reliably ranks that document first. That makes the
    expected metrics exact rather than merely plausible.
    """
    return EvalDataset.model_validate(
        {
            "name": "tiny-eval",
            "version": 1,
            "documents": [
                {
                    "key": "alpha",
                    "title": "Alpha Handbook",
                    "filename": "alpha.md",
                    "content": (
                        "# Alpha Handbook\n\n"
                        "Klaxon zebra laptop provisioning checklist for new engineers."
                    ),
                },
                {
                    "key": "beta",
                    "title": "Beta Handbook",
                    "filename": "beta.md",
                    "content": (
                        "# Beta Handbook\n\n"
                        "Quokka receipts reimbursement mileage policy for travel."
                    ),
                },
            ],
            "questions": [
                {
                    "id": "alpha-1",
                    "question": "klaxon zebra laptop provisioning",
                    "relevant_documents": ["alpha"],
                },
                {
                    "id": "alpha-2",
                    "question": "zebra laptop provisioning checklist",
                    "relevant_documents": ["alpha"],
                },
                {
                    "id": "beta-1",
                    "question": "quokka receipts reimbursement mileage",
                    "relevant_documents": ["beta"],
                },
            ],
        }
    )


async def test_evaluate_dataset_reports_metrics_and_cleans_up(
    test_settings: Any,
    database: Any,
    vector_store: Any,
    embedder: Any,
    create_tenant: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = await create_tenant("keep-me", "Keep Me")
    captured: dict[str, Tenant] = {}
    original = harness._create_eval_tenant

    async def spy(database_: Any, name: str) -> Tenant:
        tenant = await original(database_, name)
        captured["tenant"] = tenant
        return tenant

    monkeypatch.setattr(harness, "_create_eval_tenant", spy)

    report = await evaluate_dataset(
        _dataset(),
        settings=test_settings,
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
    )

    assert report.dataset.name == "tiny-eval"
    assert report.dataset.document_count == 2
    assert report.dataset.chunk_count == 2
    assert report.dataset.question_count == 3
    assert report.config.top_k == 2

    # Every question has its document ranked first, so one of the two returned
    # chunks is relevant and one is not.
    assert report.metrics.recall_at_k == 1.0
    assert report.metrics.precision_at_k == pytest.approx(0.5)
    assert report.metrics.mrr == 1.0
    assert report.metrics.hit_rate == 1.0
    assert report.metrics.irrelevant_context_rate == pytest.approx(0.5)
    assert all(
        result.retrieved[0].document_key == result.expected_documents[0]
        for result in report.per_question
    )
    assert any("target is 30-50" in warning for warning in report.warnings)

    eval_tenant = captured["tenant"]
    assert vector_store.count(str(eval_tenant.id)) == 0
    async with database.session() as session:
        remaining = (await session.execute(select(Tenant))).scalars().all()
    assert [tenant.slug for tenant in remaining] == [kept.slug]


async def test_evaluate_dataset_compares_with_a_saved_baseline(
    test_settings: Any,
    database: Any,
    vector_store: Any,
    embedder: Any,
) -> None:
    dataset = _dataset()
    baseline = await evaluate_dataset(
        dataset,
        settings=test_settings,
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
    )
    current = await evaluate_dataset(
        dataset,
        settings=test_settings,
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
        baseline=baseline,
    )

    comparisons = {item.metric: item for item in current.comparison}
    assert set(comparisons) == set(type(current.metrics).model_fields)
    assert comparisons["recall_at_k"].delta == pytest.approx(0.0)
    assert comparisons["precision_at_k"].delta == pytest.approx(0.0)
    assert comparisons["irrelevant_context_rate"].higher_is_better is False
    assert not any("fingerprint" in warning for warning in current.warnings)
    assert not any("configuration differs" in warning for warning in current.warnings)

    stale = baseline.model_copy(
        update={
            "dataset": baseline.dataset.model_copy(update={"fingerprint": "stale"}),
            "config": baseline.config.model_copy(update={"top_k": 7}),
        }
    )
    warnings = harness._baseline_warnings(stale, current.dataset.fingerprint, current.config)
    assert any("different dataset fingerprint" in warning for warning in warnings)
    assert any("top_k" in warning for warning in warnings)


async def test_top_k_must_be_positive(
    test_settings: Any, database: Any, vector_store: Any, embedder: Any
) -> None:
    with pytest.raises(EvaluationError, match="top_k"):
        await evaluate_dataset(
            _dataset(),
            settings=test_settings,
            database=database,
            vector_store=vector_store,
            embedder=embedder,
            top_k=0,
        )


async def test_run_evaluation_builds_its_own_clients(test_settings: Any, tmp_path: Path) -> None:
    from rag_engine.evaluation.cli import run_evaluation

    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        _dataset().model_dump_json(),
        encoding="utf-8",
    )

    report = await run_evaluation(
        dataset_path=dataset,
        settings=test_settings,
        top_k=2,
        min_score=0.0,
        baseline_path=None,
    )

    assert report.dataset.question_count == 3
    assert report.metrics.hit_rate == 1.0


def _variant(test_settings: Any, **updates: Any) -> Any:
    return type(test_settings).model_validate({**test_settings.model_dump(), **updates})


async def test_hybrid_report_records_fusion_and_compares_with_dense(
    test_settings: Any,
    database: Any,
    vector_store: Any,
    embedder: Any,
) -> None:
    dataset = _dataset()
    dense = await evaluate_dataset(
        dataset,
        settings=_variant(test_settings, retrieval_mode="dense"),
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
    )
    hybrid = await evaluate_dataset(
        dataset,
        settings=_variant(
            test_settings,
            retrieval_mode="hybrid",
            hybrid_dense_weight=1.0,
            hybrid_keyword_weight=1.0,
            hybrid_rrf_k=60,
            retrieval_keyword_candidates=200,
        ),
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
        baseline=dense,
    )

    # A dense run stays comparable with the Phase 5 baseline: no fusion fields.
    assert dense.config.retriever == "dense"
    assert dense.config.hybrid_dense_weight is None
    assert dense.config.keyword_candidates is None

    assert hybrid.config.retriever == "hybrid"
    assert hybrid.config.hybrid_dense_weight == 1.0
    assert hybrid.config.hybrid_keyword_weight == 1.0
    assert hybrid.config.hybrid_rrf_k == 60
    assert hybrid.config.keyword_candidates == 200

    # The comparison is against a different retriever on purpose; the report
    # has to say so rather than presenting it as like-for-like.
    assert any("baseline configuration differs" in warning for warning in hybrid.warnings)
    assert any("baseline used retriever 'dense'" in warning for warning in hybrid.warnings)
    assert hybrid.metrics.hit_rate == 1.0
    assert hybrid.metrics.recall_at_k == pytest.approx(1.0)

    markdown = render_markdown(hybrid)
    assert "- Retriever: hybrid" in markdown
    assert "Fusion: weighted reciprocal rank fusion" in markdown
    assert "keyword candidates: 200" in markdown
    assert "| Baseline |" in markdown


async def test_lexical_rerank_records_cost_separately_from_hybrid(
    test_settings: Any,
    database: Any,
    vector_store: Any,
    embedder: Any,
) -> None:
    dataset = _dataset()
    hybrid = await evaluate_dataset(
        dataset,
        settings=_variant(test_settings, retrieval_mode="hybrid"),
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
    )
    reranked = await evaluate_dataset(
        dataset,
        settings=_variant(
            test_settings,
            retrieval_mode="hybrid",
            rerank_provider="lexical",
            rerank_candidates=20,
            rerank_timeout_seconds=5.0,
        ),
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
        baseline=hybrid,
    )

    assert reranked.config.reranker == "lexical"
    assert reranked.config.rerank_model is None
    assert reranked.config.rerank_candidates == 20
    assert reranked.config.rerank_timeout_seconds == 5.0

    # Cost is measured, not inferred: every question runs the second pass.
    assert all(result.rerank_latency_ms > 0.0 for result in reranked.per_question)
    assert all(result.rerank_fallback is False for result in reranked.per_question)
    assert 0.0 < reranked.metrics.rerank_latency_mean_ms < reranked.metrics.latency_mean_ms

    # The comparison is deliberately across rerankers: the re-ranker fields are
    # explained by one dedicated warning rather than a generic field list.
    assert any("baseline used reranker 'off'" in warning for warning in reranked.warnings)
    assert reranked.metrics.hit_rate == 1.0

    markdown = render_markdown(reranked)
    assert "- Re-ranker: lexical;" in markdown
    assert "Re-rank fallbacks: 0 of 3 questions" in markdown


async def test_rerank_failures_fall_back_and_are_reported(
    test_settings: Any,
    database: Any,
    vector_store: Any,
    embedder: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingReranker:
        name = "failing"

        async def warmup(self) -> None:
            pass

        async def rerank(self, **_kwargs: Any) -> list[Any]:
            raise RerankTransientError("provider is down")

        async def close(self) -> None:
            pass

    monkeypatch.setattr(harness, "build_reranker", lambda _settings: FailingReranker())

    report = await evaluate_dataset(
        _dataset(),
        settings=_variant(test_settings, rerank_provider="lexical"),
        database=database,
        vector_store=vector_store,
        embedder=embedder,
        top_k=2,
    )

    # Fallback keeps the un-reranked order: quality is unchanged, and the
    # report records that re-ranking did not actually happen.
    assert report.metrics.hit_rate == 1.0
    assert all(result.rerank_fallback for result in report.per_question)
    assert all(result.rerank_latency_ms > 0.0 for result in report.per_question)
    assert "Re-rank fallbacks: 3 of 3 questions" in render_markdown(report)


async def test_a_reranker_that_cannot_warm_up_aborts_the_run(
    test_settings: Any,
    database: Any,
    vector_store: Any,
    embedder: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Unavailable:
        name = "unavailable"

        async def warmup(self) -> None:
            raise RerankUnavailableError("missing extra")

        async def rerank(self, **_kwargs: Any) -> list[Any]:
            raise AssertionError("should not be called")

        async def close(self) -> None:
            pass

    monkeypatch.setattr(harness, "build_reranker", lambda _settings: Unavailable())

    with pytest.raises(EvaluationError, match="re-ranker unavailable: missing extra"):
        await evaluate_dataset(
            _dataset(),
            settings=_variant(test_settings, rerank_provider="lexical"),
            database=database,
            vector_store=vector_store,
            embedder=embedder,
            top_k=2,
        )
