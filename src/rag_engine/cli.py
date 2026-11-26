"""Operator CLI.

Tenants have no self-service signup yet, so creating one is an explicit
administrative action::

    rag-engine create-tenant --slug acme --name "Acme Corp"
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from rag_engine.core.config import Settings, get_settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.core.logging import configure_logging
from rag_engine.core.models import Tenant
from rag_engine.evaluation.cli import run_command as run_evaluation_command
from rag_engine.ingestion import repository


async def _create_tenant(slug: str, name: str) -> int:
    settings = get_settings()
    database = PostgresDatabase(settings)
    try:
        async with database.session() as session:
            try:
                tenant = await repository.create_tenant(session, slug=slug, name=name)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                print(f"tenant {slug!r} already exists")
                return 1
        print(f"created tenant {tenant.slug} ({tenant.id})")
        return 0
    finally:
        await database.dispose()


async def _list_tenants() -> int:
    settings = get_settings()
    database = PostgresDatabase(settings)
    try:
        async with database.session() as session:
            tenants = (await session.execute(select(Tenant).order_by(Tenant.slug))).scalars().all()
        if not tenants:
            print("no tenants")
            return 0
        for tenant in tenants:
            print(f"{tenant.slug}\t{tenant.name}\t{tenant.id}")
        return 0
    finally:
        await database.dispose()


async def _run_benchmark_command(args: argparse.Namespace, settings: Settings) -> int:
    from rag_engine.embedding import build_embedder
    from rag_engine.evaluation.dataset import load_dataset
    from rag_engine.evaluation.load_test import (
        render_load_test_markdown,
        run_performance_suite,
    )
    from rag_engine.vectorstore import build_vector_store

    dataset = load_dataset(args.dataset)
    database = PostgresDatabase(settings)
    vector_store = build_vector_store(settings)
    embedder = build_embedder(settings)

    try:
        report = await run_performance_suite(
            dataset,
            settings=settings,
            database=database,
            vector_store=vector_store,
            embedder=embedder,
        )
        rendered = render_load_test_markdown(report)
        print(rendered)

        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        if args.markdown_output:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(rendered, encoding="utf-8")
        return 0
    finally:
        await embedder.close()
        await vector_store.close()
        await database.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-engine", description="RAG Knowledge Engine admin CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-tenant", help="create a tenant")
    create.add_argument("--slug", required=True, help="lowercase URL-safe identifier")
    create.add_argument("--name", required=True, help="human-readable tenant name")

    subparsers.add_parser("list-tenants", help="list tenants")

    evaluate = subparsers.add_parser("evaluate", help="run the retrieval evaluation harness")
    evaluate.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/datasets/enterprise_baseline.json"),
        help="dataset manifest (default: %(default)s)",
    )
    evaluate.add_argument("--top-k", type=int, default=None, help="override retrieval top_k")
    evaluate.add_argument(
        "--min-score", type=float, default=None, help="override the retrieval score floor"
    )
    evaluate.add_argument(
        "--retriever",
        choices=("dense", "hybrid"),
        default=None,
        help="override the retrieval mode (default: RAG_RETRIEVAL_MODE)",
    )
    evaluate.add_argument(
        "--dense-weight",
        type=float,
        default=None,
        help="override the dense weight used by reciprocal rank fusion",
    )
    evaluate.add_argument(
        "--keyword-weight",
        type=float,
        default=None,
        help="override the keyword (BM25) weight used by reciprocal rank fusion",
    )
    evaluate.add_argument(
        "--rrf-k", type=int, default=None, help="override the reciprocal rank fusion constant"
    )
    evaluate.add_argument(
        "--keyword-candidates",
        type=int,
        default=None,
        help="override how many PostgreSQL full-text candidates BM25 re-ranks",
    )
    evaluate.add_argument(
        "--reranker",
        choices=("off", "cohere", "cross-encoder", "lexical"),
        default=None,
        help="override the re-ranker (default: RAG_RERANK_PROVIDER)",
    )
    evaluate.add_argument(
        "--rerank-model",
        default=None,
        help="override the re-ranker model id or local model path",
    )
    evaluate.add_argument(
        "--rerank-candidates",
        type=int,
        default=None,
        help="override the candidate pool handed to the re-ranker",
    )
    evaluate.add_argument(
        "--rerank-timeout",
        type=float,
        default=None,
        dest="rerank_timeout",
        help="override the re-ranker deadline in seconds",
    )
    evaluate.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="previous report JSON to compare against",
    )
    evaluate.add_argument(
        "--save-baseline",
        type=Path,
        default=None,
        help="write this report as a new baseline JSON",
    )
    evaluate.add_argument(
        "--json-output", type=Path, default=None, help="write the full report as JSON"
    )
    evaluate.add_argument(
        "--markdown-output", type=Path, default=None, help="write the rendered report as Markdown"
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="run retrieval performance and concurrency load tests"
    )
    benchmark.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/datasets/enterprise_baseline.json"),
        help="dataset manifest (default: %(default)s)",
    )
    benchmark.add_argument(
        "--json-output", type=Path, default=None, help="write the full report as JSON"
    )
    benchmark.add_argument(
        "--markdown-output", type=Path, default=None, help="write the rendered report as Markdown"
    )

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level, log_format=settings.log_format.value)

    if args.command == "create-tenant":
        return asyncio.run(_create_tenant(args.slug, args.name))
    if args.command == "evaluate":
        return run_evaluation_command(args, settings)
    if args.command == "benchmark":
        return asyncio.run(_run_benchmark_command(args, settings))
    return asyncio.run(_list_tenants())


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
