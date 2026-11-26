"""Metric definitions: retrieval scoring, aggregation and comparison."""

from __future__ import annotations

import pytest

from rag_engine.evaluation.dataset import EvalQuestion
from rag_engine.evaluation.metrics import (
    RetrievedItem,
    aggregate_metrics,
    compare_metrics,
    evaluate_question,
    percentile,
)


def _item(key: str, index: int = 0, score: float = 0.9) -> RetrievedItem:
    return RetrievedItem(document_key=key, chunk_index=index, score=score)


def _question(*expected: str, question_id: str = "q1") -> EvalQuestion:
    return EvalQuestion(
        id=question_id,
        question="What does the policy say?",
        relevant_documents=list(expected),
    )


def test_evaluate_question_counts_unique_documents_for_recall() -> None:
    result = evaluate_question(
        _question("alpha", "beta"),
        [_item("alpha", 0), _item("alpha", 1), _item("beta", 0), _item("gamma", 0)],
        k=5,
        latency_ms=12.5,
    )

    assert result.recall == 1.0
    assert result.precision == pytest.approx(3 / 5)
    assert result.reciprocal_rank == 1.0
    assert result.relevant_retrieved == 3
    assert result.irrelevant_context_rate == pytest.approx(1 / 4)
    assert result.latency_ms == 12.5


def test_evaluate_question_uses_k_as_the_precision_denominator() -> None:
    result = evaluate_question(
        _question("alpha"),
        [_item("alpha")],
        k=5,
        latency_ms=1.0,
    )

    assert result.recall == 1.0
    assert result.precision == pytest.approx(1 / 5)
    assert result.irrelevant_context_rate == 0.0


def test_evaluate_question_scores_only_the_top_k() -> None:
    retrieved = [
        _item("noise", 0),
        _item("noise", 1),
        _item("noise", 2),
        _item("alpha", 0),
    ]

    result = evaluate_question(_question("alpha"), retrieved, k=3, latency_ms=1.0)

    assert result.recall == 0.0
    assert result.precision == 0.0
    assert result.reciprocal_rank == 0.0
    assert len(result.retrieved) == 3


def test_evaluate_question_finds_the_first_relevant_rank() -> None:
    result = evaluate_question(
        _question("alpha"),
        [_item("noise"), _item("other"), _item("alpha")],
        k=3,
        latency_ms=1.0,
    )

    assert result.reciprocal_rank == pytest.approx(1 / 3)


def test_evaluate_question_partial_recall_and_empty_results() -> None:
    partial = evaluate_question(
        _question("alpha", "beta"),
        [_item("alpha")],
        k=2,
        latency_ms=1.0,
    )
    empty = evaluate_question(_question("alpha"), [], k=2, latency_ms=1.0)

    assert partial.recall == 0.5
    assert empty.recall == 0.0
    assert empty.precision == 0.0
    assert empty.irrelevant_context_rate == 0.0


def test_percentile_uses_nearest_rank() -> None:
    values = list(range(1, 21))

    assert percentile(values, 50.0) == 10.0
    assert percentile(values, 95.0) == 19.0
    assert percentile(values, 100.0) == 20.0
    assert percentile([], 95.0) == 0.0


def test_percentile_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 0.0)
    with pytest.raises(ValueError):
        percentile([1.0], 101.0)


def test_aggregate_metrics_averages_questions_and_micro_averages_context() -> None:
    # The first question returns fewer than k chunks, so the micro-averaged
    # irrelevant-context rate (2/5) differs from the mean of per-question rates
    # ((0.5 + 1/3) / 2), which is the point of micro-averaging.
    first = evaluate_question(
        _question("alpha", question_id="q1"),
        [_item("alpha"), _item("noise")],
        k=3,
        latency_ms=10.0,
    )
    second = evaluate_question(
        _question("alpha", "beta", question_id="q2"),
        [_item("alpha"), _item("beta"), _item("noise")],
        k=3,
        latency_ms=20.0,
    )

    metrics = aggregate_metrics([first, second], k=3)

    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == pytest.approx(0.5)
    assert metrics.mrr == 1.0
    assert metrics.hit_rate == 1.0
    assert metrics.irrelevant_context_rate == pytest.approx(0.4)
    assert metrics.latency_p50_ms == 10.0
    assert metrics.latency_p95_ms == 20.0
    assert metrics.latency_mean_ms == 15.0


def test_aggregate_metrics_empty_input_is_an_error() -> None:
    with pytest.raises(ValueError):
        aggregate_metrics([], k=5)


def test_aggregate_metrics_reports_rerank_latency_separately() -> None:
    first = evaluate_question(
        _question("alpha", question_id="q1"),
        [_item("alpha")],
        k=1,
        latency_ms=30.0,
        rerank_latency_ms=8.0,
    )
    second = evaluate_question(
        _question("alpha", question_id="q2"),
        [_item("alpha")],
        k=1,
        latency_ms=40.0,
        rerank_latency_ms=12.0,
        rerank_fallback=True,
    )

    metrics = aggregate_metrics([first, second], k=1)

    assert metrics.rerank_latency_p50_ms == 8.0
    assert metrics.rerank_latency_p95_ms == 12.0
    assert metrics.rerank_latency_mean_ms == 10.0
    # Re-rank latency is a share of the total, never an addition to it.
    assert metrics.latency_mean_ms == 35.0


def test_rerank_defaults_keep_pre_rerank_results_compatible() -> None:
    result = evaluate_question(
        _question("alpha"),
        [_item("alpha")],
        k=1,
        latency_ms=10.0,
    )

    assert result.rerank_latency_ms == 0.0
    assert result.rerank_fallback is False
    metrics = aggregate_metrics([result], k=1)
    assert metrics.rerank_latency_mean_ms == 0.0


def test_compare_metrics_records_signed_deltas_and_directions() -> None:
    result = evaluate_question(
        _question("alpha"),
        [_item("alpha")],
        k=1,
        latency_ms=10.0,
    )
    current = aggregate_metrics([result], k=1)
    baseline = current.model_copy(update={"recall_at_k": 0.5, "latency_p50_ms": 20.0})

    comparisons = {item.metric: item for item in compare_metrics(current, baseline)}

    assert comparisons["recall_at_k"].delta == pytest.approx(0.5)
    assert comparisons["recall_at_k"].higher_is_better is True
    assert comparisons["latency_p50_ms"].delta == pytest.approx(-10.0)
    assert comparisons["latency_p50_ms"].higher_is_better is False
    assert set(comparisons) == set(type(current).model_fields)
