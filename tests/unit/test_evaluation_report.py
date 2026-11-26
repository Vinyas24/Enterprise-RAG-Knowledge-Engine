"""Report rendering: the human-readable artifact a reviewer reads."""

from __future__ import annotations

from datetime import UTC, datetime

from rag_engine.evaluation.dataset import EvalQuestion
from rag_engine.evaluation.metrics import (
    RetrievedItem,
    aggregate_metrics,
    compare_metrics,
    evaluate_question,
)
from rag_engine.evaluation.report import (
    DatasetSummary,
    EvaluationConfig,
    EvaluationReport,
    render_markdown,
)


def _report(
    *, baseline: bool = False, warning: str | None = None, reranker: str = "off"
) -> EvaluationReport:
    reranking = reranker != "off"
    question = EvalQuestion(
        id="q1",
        question="How do I roll back?",
        relevant_documents=["deployment-runbook"],
    )
    hit = evaluate_question(
        question,
        [RetrievedItem(document_key="deployment-runbook", chunk_index=0, score=0.9)],
        k=5,
        latency_ms=10.0,
        rerank_latency_ms=4.0 if reranking else 0.0,
    )
    miss = evaluate_question(
        question.model_copy(update={"id": "q2", "question": "Something unanswered"}),
        [RetrievedItem(document_key="noise", chunk_index=0, score=0.1)],
        k=5,
        latency_ms=30.0,
        rerank_latency_ms=6.0 if reranking else 0.0,
        rerank_fallback=reranking,
    )
    results = [hit, miss]
    metrics = aggregate_metrics(results, k=5)
    config = EvaluationConfig(
        embedding_provider="deterministic",
        embedding_dimensions=256,
        chunk_size_chars=1200,
        chunk_overlap_chars=200,
        vector_store="memory",
        top_k=5,
        min_score=0.0,
        reranker=reranker,
        rerank_model="rerank-v3.5" if reranker == "cohere" else None,
        rerank_candidates=20 if reranking else None,
        rerank_timeout_seconds=5.0 if reranking else None,
    )
    return EvaluationReport(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset=DatasetSummary(
            name="tiny",
            version=1,
            fingerprint="abc123",
            document_count=1,
            chunk_count=2,
            question_count=2,
        ),
        config=config,
        metrics=metrics,
        per_question=results,
        comparison=(
            compare_metrics(metrics, metrics.model_copy(update={"recall_at_k": 0.5}))
            if baseline
            else []
        ),
        warnings=[warning] if warning else [],
    )


def test_markdown_states_the_dataset_configuration_and_definitions() -> None:
    markdown = render_markdown(_report())

    assert "# Retrieval evaluation — tiny" in markdown
    assert "Dataset fingerprint: `abc123`" in markdown
    assert "- Questions: 2" in markdown
    assert "top_k: 5" in markdown
    assert "| Recall@k |" in markdown
    assert "Relevance is defined" not in markdown
    assert "A retrieved chunk is **relevant**" in markdown
    assert "`Retriever.retrieve` only" in markdown
    assert "nearest-rank" in markdown


def test_markdown_lists_warnings_and_misses() -> None:
    markdown = render_markdown(_report(warning="dataset is small"))

    assert "## Warnings" in markdown
    assert "- dataset is small" in markdown
    assert "## Misses and partial recalls" in markdown
    assert "`q2` Something unanswered" in markdown


def test_markdown_shows_baseline_values_and_directions() -> None:
    markdown = render_markdown(_report(baseline=True))

    assert "| Baseline |" in markdown
    assert "| 0.5000 |" in markdown
    assert "lower is better" in markdown
    assert "higher is better" in markdown


def test_report_json_round_trips() -> None:
    report = _report(baseline=True, warning="dataset is small")

    restored = EvaluationReport.model_validate_json(report.model_dump_json())

    assert restored == report


def test_markdown_hides_rerank_rows_when_reranking_is_disabled() -> None:
    markdown = render_markdown(_report())

    assert "| Re-rank latency" not in markdown
    assert "Re-ranker:" not in markdown
    assert "Re-rank fallbacks" not in markdown
    assert "neither BM25 fusion nor a re-ranker is applied" in markdown


def test_markdown_reports_rerank_cost_configuration_and_fallbacks() -> None:
    markdown = render_markdown(_report(reranker="cohere"))

    assert "Re-ranker: cohere (rerank-v3.5); 20 candidates → top 5, timeout 5s" in markdown
    assert "| Re-rank latency p50 (ms) | 4.0000 |" in markdown
    assert "| Re-rank latency p95 (ms) | 6.0000 |" in markdown
    assert "Re-rank fallbacks: 1 of 2 questions" in markdown
    assert "then cohere re-ranking" in markdown
    assert "re-ranker relevance scores" in markdown


def test_rerank_defaults_do_not_break_an_old_baseline() -> None:
    """A Phase 5/6 JSON has no rerank fields; it must still load and compare."""
    legacy = _report().model_dump(mode="json")
    for field in (
        "reranker",
        "rerank_model",
        "rerank_candidates",
        "rerank_timeout_seconds",
    ):
        legacy["config"].pop(field)
    for result in legacy["per_question"]:
        result.pop("rerank_latency_ms")
        result.pop("rerank_fallback")
    legacy["metrics"].pop("rerank_latency_p50_ms")
    legacy["metrics"].pop("rerank_latency_p95_ms")
    legacy["metrics"].pop("rerank_latency_mean_ms")

    restored = EvaluationReport.model_validate(legacy)

    assert restored.config.reranker == "off"
    assert restored.metrics.rerank_latency_mean_ms == 0.0
