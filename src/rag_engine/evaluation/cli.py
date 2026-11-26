"""Command-line entry point for the retrieval evaluation harness."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rag_engine.core.config import Settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.embedding import build_embedder
from rag_engine.vectorstore import build_vector_store

from .dataset import load_dataset
from .harness import EvaluationError, evaluate_dataset
from .report import EvaluationReport, render_markdown


async def run_evaluation(
    *,
    dataset_path: Path,
    settings: Settings,
    top_k: int | None,
    min_score: float | None,
    baseline_path: Path | None,
) -> EvaluationReport:
    """Load inputs, build clients, evaluate, and always close the clients."""
    dataset = load_dataset(dataset_path)
    baseline = _load_report(baseline_path) if baseline_path is not None else None

    database = PostgresDatabase(settings)
    vector_store = build_vector_store(settings)
    embedder = build_embedder(settings)
    try:
        await vector_store.ensure_ready()
        return await evaluate_dataset(
            dataset,
            settings=settings,
            database=database,
            vector_store=vector_store,
            embedder=embedder,
            top_k=top_k,
            min_score=min_score,
            baseline=baseline,
        )
    finally:
        await embedder.close()
        await vector_store.close()
        await database.dispose()


def _load_report(path: Path) -> EvaluationReport:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read baseline {path}: {exc}") from exc
    return EvaluationReport.model_validate_json(payload)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Return settings with CLI overrides applied and validated.

    ``model_copy`` would bypass the cross-field checks (a zero-weight pair, for
    example), so the merged values are validated as a fresh ``Settings``.
    """
    updates: dict[str, object] = {}
    if args.retriever is not None:
        updates["retrieval_mode"] = args.retriever
    if args.dense_weight is not None:
        updates["hybrid_dense_weight"] = args.dense_weight
    if args.keyword_weight is not None:
        updates["hybrid_keyword_weight"] = args.keyword_weight
    if args.rrf_k is not None:
        updates["hybrid_rrf_k"] = args.rrf_k
    if args.keyword_candidates is not None:
        updates["retrieval_keyword_candidates"] = args.keyword_candidates
    if args.reranker is not None:
        updates["rerank_provider"] = args.reranker
    if args.rerank_model is not None:
        updates["rerank_model"] = args.rerank_model
    if args.rerank_candidates is not None:
        updates["rerank_candidates"] = args.rerank_candidates
    if args.rerank_timeout is not None:
        updates["rerank_timeout_seconds"] = args.rerank_timeout
    if not updates:
        return settings
    return Settings.model_validate({**settings.model_dump(), **updates})


def run_command(args: argparse.Namespace, settings: Settings) -> int:
    """Execute the ``rag-engine evaluate`` subcommand."""
    try:
        resolved_settings = _apply_overrides(settings, args)
        report = asyncio.run(
            run_evaluation(
                dataset_path=args.dataset,
                settings=resolved_settings,
                top_k=args.top_k,
                min_score=args.min_score,
                baseline_path=args.baseline,
            )
        )
    except (EvaluationError, ValueError, OSError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    markdown = render_markdown(report)
    print(markdown)

    if args.json_output is not None:
        _write(args.json_output, report.model_dump_json(indent=2) + "\n")
    if args.markdown_output is not None:
        _write(args.markdown_output, markdown + "\n")
    if args.save_baseline is not None:
        _write(args.save_baseline, report.model_dump_json(indent=2) + "\n")
    return 0
