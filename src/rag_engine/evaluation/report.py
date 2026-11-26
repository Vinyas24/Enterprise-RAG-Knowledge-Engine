"""Evaluation report models and human-readable rendering.

The JSON form is the machine interface and the saved baseline. The Markdown
form is what a reviewer reads after a change; it always states how each metric
is defined, because a number without a definition invites false comparisons.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .metrics import AggregateMetrics, MetricComparison, QuestionResult


class DatasetSummary(BaseModel):
    """What was evaluated, including a fingerprint that detects drift."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    fingerprint: str
    document_count: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    question_count: int = Field(ge=1)


class EvaluationConfig(BaseModel):
    """Everything that can change a retrieval number."""

    model_config = ConfigDict(frozen=True)

    retriever: str = "dense"
    embedding_provider: str
    embedding_dimensions: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    vector_store: str
    top_k: int
    min_score: float
    # Hybrid-only settings. ``None`` on a dense-only run so a dense report is
    # comparable with the Phase 5 baseline without a spurious config warning.
    keyword_candidates: int | None = None
    hybrid_dense_weight: float | None = None
    hybrid_keyword_weight: float | None = None
    hybrid_rrf_k: int | None = None
    # Phase 7. Defaults describe Phase 5/6 behaviour, so a saved dense or hybrid
    # baseline still loads and compares without a reranker.
    reranker: str = "off"
    rerank_model: str | None = None
    rerank_candidates: int | None = None
    rerank_timeout_seconds: float | None = None


class EvaluationReport(BaseModel):
    """A complete, reproducible evaluation result."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: datetime
    dataset: DatasetSummary
    config: EvaluationConfig
    metrics: AggregateMetrics
    per_question: list[QuestionResult]
    comparison: list[MetricComparison] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


_METRIC_LABELS = {
    "recall_at_k": "Recall@k",
    "precision_at_k": "Precision@k",
    "mrr": "MRR",
    "hit_rate": "Hit rate",
    "irrelevant_context_rate": "Irrelevant-context rate",
    "latency_p50_ms": "Latency p50 (ms)",
    "latency_p95_ms": "Latency p95 (ms)",
    "latency_mean_ms": "Latency mean (ms)",
    "rerank_latency_p50_ms": "Re-rank latency p50 (ms)",
    "rerank_latency_p95_ms": "Re-rank latency p95 (ms)",
    "rerank_latency_mean_ms": "Re-rank latency mean (ms)",
}

_METRIC_DIRECTIONS = {
    "recall_at_k": "higher is better",
    "precision_at_k": "higher is better",
    "mrr": "higher is better",
    "hit_rate": "higher is better",
    "irrelevant_context_rate": "lower is better",
    "latency_p50_ms": "lower is better",
    "latency_p95_ms": "lower is better",
    "latency_mean_ms": "lower is better",
    "rerank_latency_p50_ms": "lower is better",
    "rerank_latency_p95_ms": "lower is better",
    "rerank_latency_mean_ms": "lower is better",
}


def _format_value(metric: str, value: float) -> str:
    if metric.startswith("latency_"):
        return f"{value:.2f}"
    return f"{value:.4f}"


def _format_delta(metric: str, delta: float) -> str:
    if metric.startswith("latency_"):
        return f"{delta:+.2f}"
    return f"{delta:+.4f}"


def render_markdown(report: EvaluationReport) -> str:
    """Render the report a reviewer sees, including metric definitions."""
    k = report.config.top_k
    comparisons = {item.metric: item for item in report.comparison}
    lines: list[str] = [
        f"# Retrieval evaluation — {report.dataset.name}",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        f"Dataset fingerprint: `{report.dataset.fingerprint}`",
        "",
        "## Corpus and configuration",
        "",
        f"- Documents: {report.dataset.document_count}",
        f"- Indexed chunks: {report.dataset.chunk_count}",
        f"- Questions: {report.dataset.question_count}",
        (
            f"- Retriever: {report.config.retriever} "
            f"({report.config.embedding_provider}, "
            f"{report.config.embedding_dimensions} dimensions)"
        ),
        (
            f"- Chunking: {report.config.chunk_size_chars} chars / "
            f"{report.config.chunk_overlap_chars} overlap"
        ),
        f"- Vector store: {report.config.vector_store}",
        f"- top_k: {k}, min_score: {report.config.min_score}",
    ]
    if report.config.hybrid_dense_weight is not None:
        lines.append(
            f"- Fusion: weighted reciprocal rank fusion "
            f"(dense {report.config.hybrid_dense_weight}, "
            f"keyword {report.config.hybrid_keyword_weight}, "
            f"k={report.config.hybrid_rrf_k}); "
            f"keyword candidates: {report.config.keyword_candidates}"
        )
    if report.config.reranker != "off":
        model = f" ({report.config.rerank_model})" if report.config.rerank_model else ""
        lines.append(
            f"- Re-ranker: {report.config.reranker}{model}; "
            f"{report.config.rerank_candidates} candidates → top {k}, "
            f"timeout {report.config.rerank_timeout_seconds:g}s"
        )
    lines.append("")

    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")

    lines.extend(
        [
            "## Metrics",
            "",
            "| Metric | Current | Baseline | Δ | Direction |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for metric, label in _METRIC_LABELS.items():
        if report.config.reranker == "off" and metric.startswith("rerank_"):
            continue
        value = getattr(report.metrics, metric)
        comparison = comparisons.get(metric)
        baseline = _format_value(metric, comparison.baseline) if comparison is not None else "—"
        delta = _format_delta(metric, comparison.delta) if comparison is not None else "—"
        lines.append(
            f"| {label} | {_format_value(metric, value)} | {baseline} | {delta} | "
            f"{_METRIC_DIRECTIONS[metric]} |"
        )
    lines.append("")

    if report.config.reranker != "off":
        fallbacks = sum(1 for result in report.per_question if result.rerank_fallback)
        lines.append(
            f"Re-rank fallbacks: {fallbacks} of {len(report.per_question)} questions returned "
            "the un-reranked order."
        )
        lines.append("")

    misses = [
        result
        for result in report.per_question
        if result.relevant_retrieved == 0 or result.recall < 1.0
    ]
    lines.extend(["## Misses and partial recalls", ""])
    if not misses:
        lines.append(
            "None — every question retrieved at least one chunk from every expected document."
        )
    else:
        for result in misses:
            retrieved = ", ".join(
                f"{item.document_key}:{item.chunk_index}" for item in result.retrieved
            )
            expected = ", ".join(result.expected_documents)
            lines.append(
                f"- `{result.question_id}` {result.question} "
                f"(expected: {expected}; recall {result.recall:.2f}; retrieved: {retrieved})"
            )
    lines.append("")

    lines.extend(
        [
            "## Metric definitions",
            "",
            (
                "A retrieved chunk is **relevant** when its document key is listed in the "
                "question's `relevant_documents`. Labels are document-level so they survive "
                "chunking changes."
            ),
            "",
            f"- **Recall@{k}**: unique expected documents found in the top {k} chunks, "
            "divided by the number of expected documents; averaged over questions.",
            f"- **Precision@{k}**: relevant chunks in the top {k}, divided by {k}; averaged "
            "over questions. Fewer than k results are not rounded up.",
            "- **MRR**: reciprocal rank of the first relevant chunk (0 when none); averaged.",
            "- **Hit rate**: share of questions with at least one relevant chunk.",
            "- **Irrelevant-context rate**: irrelevant retrieved chunks divided by all "
            "retrieved chunks, micro-averaged across questions. This is the share of "
            "context handed to the generator that the labels do not consider relevant.",
            "- **Latency**: wall time around `Retriever.retrieve` only — query embedding, "
            "vector search, keyword candidate selection and BM25 scoring, PostgreSQL "
            "hydration, clearance filtering and re-ranking when enabled, whichever of those "
            "the configured stack runs. Measured after one warm-up retrieval. It excludes "
            "ingestion, dataset loading, metric calculation, report rendering and process "
            "startup. p50/p95 use the nearest-rank method.",
            "- **Re-rank latency**: time spent inside the re-ranker, a subset of the total "
            "latency above; zero when re-ranking is disabled. A timeout bills the full "
            "deadline before the fallback returns the un-reranked order.",
            "",
            _retrieval_path(report),
            "",
        ]
    )
    return "\n".join(lines)


def _retrieval_path(report: EvaluationReport) -> str:
    """One sentence describing ordering and the score scale, honestly."""
    fused = report.config.hybrid_dense_weight is not None
    if report.config.reranker != "off":
        upstream = (
            "weighted reciprocal rank fusion of dense and BM25 keyword results"
            if fused
            else "dense search"
        )
        return (
            f"Retrieval path: {upstream}, then {report.config.reranker} re-ranking of the "
            "merged candidates down to top_k. Scores in `per_question` are re-ranker "
            "relevance scores, so they are not comparable across providers or with cosine "
            "similarities."
        )
    if fused:
        return (
            "Retrieval path: weighted reciprocal rank fusion of dense and BM25 keyword "
            "results; no re-ranker is applied. Scores in `per_question` are fused RRF "
            "scores, so they are not comparable with cosine similarities."
        )
    return "Retrieval path: dense only — neither BM25 fusion nor a re-ranker is applied."
