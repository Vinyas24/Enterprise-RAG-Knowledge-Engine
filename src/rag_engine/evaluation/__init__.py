"""Retrieval evaluation: dataset, metrics, harness and report."""

from __future__ import annotations

from .dataset import (
    EvalDataset,
    EvalDocument,
    EvalQuestion,
    dataset_fingerprint,
    load_dataset,
)
from .harness import EvaluationError, evaluate_dataset
from .metrics import (
    AggregateMetrics,
    MetricComparison,
    QuestionResult,
    RetrievedItem,
    aggregate_metrics,
    compare_metrics,
    evaluate_question,
    percentile,
)
from .report import DatasetSummary, EvaluationConfig, EvaluationReport, render_markdown

__all__ = [
    "AggregateMetrics",
    "DatasetSummary",
    "EvalDataset",
    "EvalDocument",
    "EvalQuestion",
    "EvaluationConfig",
    "EvaluationError",
    "EvaluationReport",
    "MetricComparison",
    "QuestionResult",
    "RetrievedItem",
    "aggregate_metrics",
    "compare_metrics",
    "dataset_fingerprint",
    "evaluate_dataset",
    "evaluate_question",
    "load_dataset",
    "percentile",
    "render_markdown",
]
