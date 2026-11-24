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


def run_command(args: argparse.Namespace, settings: Settings) -> int:
    """Execute the ``rag-engine evaluate`` subcommand."""
    try:
        report = asyncio.run(
            run_evaluation(
                dataset_path=args.dataset,
                settings=settings,
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
