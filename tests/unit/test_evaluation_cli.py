"""Evaluation CLI behaviour without a database."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from rag_engine.core.config import RerankProvider, RetrievalMode, Settings
from rag_engine.evaluation import cli as evaluation_cli
from rag_engine.evaluation.cli import run_command
from rag_engine.evaluation.report import EvaluationReport


def _report() -> EvaluationReport:
    return EvaluationReport.model_validate(
        {
            "generated_at": "2026-01-01T00:00:00Z",
            "dataset": {
                "name": "tiny",
                "version": 1,
                "fingerprint": "abc",
                "document_count": 1,
                "chunk_count": 1,
                "question_count": 1,
            },
            "config": {
                "embedding_provider": "deterministic",
                "embedding_dimensions": 256,
                "chunk_size_chars": 1200,
                "chunk_overlap_chars": 200,
                "vector_store": "memory",
                "top_k": 1,
                "min_score": 0.0,
            },
            "metrics": {
                "recall_at_k": 1.0,
                "precision_at_k": 1.0,
                "mrr": 1.0,
                "hit_rate": 1.0,
                "irrelevant_context_rate": 0.0,
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 1.0,
                "latency_mean_ms": 1.0,
            },
            "per_question": [
                {
                    "question_id": "q1",
                    "question": "What?",
                    "expected_documents": ["alpha"],
                    "retrieved": [{"document_key": "alpha", "chunk_index": 0, "score": 0.9}],
                    "relevant_retrieved": 1,
                    "recall": 1.0,
                    "precision": 1.0,
                    "reciprocal_rank": 1.0,
                    "irrelevant_context_rate": 0.0,
                    "latency_ms": 1.0,
                }
            ],
        }
    )


def _args(dataset: Path) -> Namespace:
    return Namespace(
        dataset=dataset,
        top_k=None,
        min_score=None,
        retriever=None,
        dense_weight=None,
        keyword_weight=None,
        rrf_k=None,
        keyword_candidates=None,
        reranker=None,
        rerank_model=None,
        rerank_candidates=None,
        rerank_timeout=None,
        baseline=None,
        save_baseline=None,
        json_output=None,
        markdown_output=None,
    )


def test_missing_dataset_returns_a_nonzero_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = run_command(_args(tmp_path / "missing.json"), Settings(_env_file=None))

    assert exit_code == 2
    assert "evaluation failed" in capsys.readouterr().err


def test_invalid_dataset_json_returns_a_nonzero_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{not json", encoding="utf-8")

    exit_code = run_command(_args(dataset), Settings(_env_file=None))

    assert exit_code == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_run_command_prints_and_writes_every_requested_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _report()

    async def fake_run_evaluation(**_kwargs: object) -> EvaluationReport:
        return expected

    monkeypatch.setattr(evaluation_cli, "run_evaluation", fake_run_evaluation)
    args = _args(tmp_path / "dataset.json")
    args.json_output = tmp_path / "report.json"
    args.markdown_output = tmp_path / "report.md"
    args.save_baseline = tmp_path / "baseline.json"

    exit_code = run_command(args, Settings(_env_file=None))

    assert exit_code == 0
    assert "# Retrieval evaluation — tiny" in capsys.readouterr().out
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# Retrieval evaluation")
    assert (
        EvaluationReport.model_validate_json(
            (tmp_path / "baseline.json").read_text(encoding="utf-8")
        )
        == expected
    )


def test_retriever_and_fusion_overrides_reach_the_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_evaluation(**kwargs: object) -> EvaluationReport:
        captured.update(kwargs)
        return _report()

    monkeypatch.setattr(evaluation_cli, "run_evaluation", fake_run_evaluation)
    args = _args(tmp_path / "dataset.json")
    args.retriever = "dense"
    args.dense_weight = 2.0
    args.keyword_weight = 0.25
    args.rrf_k = 30
    args.keyword_candidates = 50

    assert run_command(args, Settings(_env_file=None)) == 0

    settings = captured["settings"]
    assert isinstance(settings, Settings)
    assert settings.retrieval_mode is RetrievalMode.DENSE
    assert settings.hybrid_dense_weight == 2.0
    assert settings.hybrid_keyword_weight == 0.25
    assert settings.hybrid_rrf_k == 30
    assert settings.retrieval_keyword_candidates == 50


def test_zero_fusion_weights_are_rejected_before_any_database_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(tmp_path / "dataset.json")
    args.dense_weight = 0.0
    args.keyword_weight = 0.0

    exit_code = run_command(args, Settings(_env_file=None))

    assert exit_code == 2
    assert "cannot both be zero" in capsys.readouterr().err


def test_rerank_overrides_reach_the_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_evaluation(**kwargs: object) -> EvaluationReport:
        captured.update(kwargs)
        return _report()

    monkeypatch.setattr(evaluation_cli, "run_evaluation", fake_run_evaluation)
    args = _args(tmp_path / "dataset.json")
    args.reranker = "cross-encoder"
    args.rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    args.rerank_candidates = 30
    args.rerank_timeout = 2.5

    assert run_command(args, Settings(_env_file=None)) == 0

    settings = captured["settings"]
    assert isinstance(settings, Settings)
    assert settings.rerank_provider is RerankProvider.CROSS_ENCODER
    assert settings.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.rerank_candidates == 30
    assert settings.rerank_timeout_seconds == 2.5


def test_rerank_override_without_a_cohere_key_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(tmp_path / "dataset.json")
    args.reranker = "cohere"

    exit_code = run_command(args, Settings(_env_file=None))

    assert exit_code == 2
    assert "rerank_api_key is required" in capsys.readouterr().err
