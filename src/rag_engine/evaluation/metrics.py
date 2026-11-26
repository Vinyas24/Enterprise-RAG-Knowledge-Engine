"""Retrieval metrics with explicit, reproducible definitions.

Relevance is defined at document level: a retrieved chunk is relevant when its
document key is one of the question's ``relevant_documents``. That makes labels
stable across chunking changes, which is what lets a later chunking experiment
be compared on the same dataset.

All rate metrics are averaged per question unless stated otherwise. The
irrelevant-context rate is micro-averaged over every retrieved chunk, because
the question "how much of what we put in the prompt is irrelevant?" is about
the prompt as a whole, not about the average question.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .dataset import EvalQuestion


class RetrievedItem(BaseModel):
    """One retrieved chunk, mapped back to its dataset document key."""

    model_config = ConfigDict(frozen=True)

    document_key: str
    chunk_index: int = Field(ge=0)
    score: float


class QuestionResult(BaseModel):
    """Metrics and evidence for one question."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    expected_documents: list[str]
    retrieved: list[RetrievedItem]
    relevant_retrieved: int = Field(ge=0)
    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    irrelevant_context_rate: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    # The re-rank share of ``latency_ms``. Zero when re-ranking is disabled, so
    # a Phase 5/6 baseline JSON still parses and compares.
    rerank_latency_ms: float = Field(default=0.0, ge=0.0)
    rerank_fallback: bool = False


class AggregateMetrics(BaseModel):
    """The numbers compared between runs."""

    model_config = ConfigDict(frozen=True)

    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    irrelevant_context_rate: float = Field(ge=0.0, le=1.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_mean_ms: float = Field(ge=0.0)
    # Re-ranking cost, a subset of the totals above. Defaults keep old
    # baselines loadable and render as a zero-cost baseline.
    rerank_latency_p50_ms: float = Field(default=0.0, ge=0.0)
    rerank_latency_p95_ms: float = Field(default=0.0, ge=0.0)
    rerank_latency_mean_ms: float = Field(default=0.0, ge=0.0)


class MetricComparison(BaseModel):
    """One baseline/current pair in a comparison report."""

    model_config = ConfigDict(frozen=True)

    metric: str
    baseline: float
    current: float
    delta: float
    higher_is_better: bool


# Latency and irrelevant context are the metrics where a decrease is good.
_LOWER_IS_BETTER = {
    "irrelevant_context_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_mean_ms",
    "rerank_latency_p50_ms",
    "rerank_latency_p95_ms",
    "rerank_latency_mean_ms",
}


def evaluate_question(
    question: EvalQuestion,
    retrieved: Sequence[RetrievedItem],
    *,
    k: int,
    latency_ms: float,
    rerank_latency_ms: float = 0.0,
    rerank_fallback: bool = False,
) -> QuestionResult:
    """Score one question's top-k results.

    ``recall`` counts unique expected documents found, so retrieving three
    chunks from the right document does not inflate recall past 1.0.
    ``precision`` uses the configured ``k`` as the denominator, including when
    the backend returned fewer than ``k`` results.
    """
    expected = set(question.relevant_documents)
    top = list(retrieved[:k])
    relevant = [item for item in top if item.document_key in expected]

    recall = len({item.document_key for item in relevant}) / len(expected)
    precision = len(relevant) / k
    reciprocal_rank = 0.0
    for rank, item in enumerate(top, start=1):
        if item.document_key in expected:
            reciprocal_rank = 1.0 / rank
            break
    irrelevant = len(top) - len(relevant)
    irrelevant_rate = irrelevant / len(top) if top else 0.0

    return QuestionResult(
        question_id=question.id,
        question=question.question,
        expected_documents=list(question.relevant_documents),
        retrieved=top,
        relevant_retrieved=len(relevant),
        recall=recall,
        precision=precision,
        reciprocal_rank=reciprocal_rank,
        irrelevant_context_rate=irrelevant_rate,
        latency_ms=latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        rerank_fallback=rerank_fallback,
    )


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """Nearest-rank percentile.

    With few samples this is more honest than interpolation: p95 of 20 values
    is the 19th value, not a synthetic point between the last two.
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if not 0.0 < percentile_value <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    rank = math.ceil(percentile_value / 100.0 * len(ordered))
    return ordered[max(0, rank - 1)]


def aggregate_metrics(results: Sequence[QuestionResult], *, k: int) -> AggregateMetrics:
    """Aggregate per-question results into the report's headline numbers."""
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    count = len(results)
    latencies = [result.latency_ms for result in results]
    rerank_latencies = [result.rerank_latency_ms for result in results]
    retrieved_total = sum(len(result.retrieved[:k]) for result in results)
    relevant_total = sum(result.relevant_retrieved for result in results)

    return AggregateMetrics(
        recall_at_k=sum(result.recall for result in results) / count,
        precision_at_k=sum(result.precision for result in results) / count,
        mrr=sum(result.reciprocal_rank for result in results) / count,
        hit_rate=sum(1 for result in results if result.relevant_retrieved > 0) / count,
        irrelevant_context_rate=(
            1.0 - relevant_total / retrieved_total if retrieved_total else 0.0
        ),
        latency_p50_ms=percentile(latencies, 50.0),
        latency_p95_ms=percentile(latencies, 95.0),
        latency_mean_ms=sum(latencies) / count,
        rerank_latency_p50_ms=percentile(rerank_latencies, 50.0),
        rerank_latency_p95_ms=percentile(rerank_latencies, 95.0),
        rerank_latency_mean_ms=sum(rerank_latencies) / count,
    )


def compare_metrics(
    current: AggregateMetrics, baseline: AggregateMetrics
) -> list[MetricComparison]:
    """Pair every metric with its baseline value and signed delta."""
    comparisons: list[MetricComparison] = []
    for metric in type(current).model_fields:
        current_value = getattr(current, metric)
        baseline_value = getattr(baseline, metric)
        comparisons.append(
            MetricComparison(
                metric=metric,
                baseline=baseline_value,
                current=current_value,
                delta=current_value - baseline_value,
                higher_is_better=metric not in _LOWER_IS_BETTER,
            )
        )
    return comparisons
