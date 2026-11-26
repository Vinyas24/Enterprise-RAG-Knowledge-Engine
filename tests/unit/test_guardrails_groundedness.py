"""Unit tests for output groundedness checker."""

from __future__ import annotations

from uuid import uuid4

from rag_engine.core.enums import AccessLevel
from rag_engine.generation.prompts import NOT_FOUND_RESPONSE
from rag_engine.guardrails.groundedness import check_groundedness
from rag_engine.retrieval.base import RetrievedChunk


def _make_chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title="Kubernetes Guide",
        chunk_index=0,
        content=content,
        score=0.9,
        access_level=AccessLevel.INTERNAL,
    )


def test_grounded_answer_passes() -> None:
    passages = [
        _make_chunk(
            "To roll back a deployment, run kubectl rollout undo deployment/api-server. "
            "You can check status with kubectl rollout status."
        )
    ]
    answer = (
        "Run kubectl rollout undo deployment/api-server to roll back. "
        "Check status with kubectl rollout status [1]."
    )

    result = check_groundedness(answer, passages, min_score=0.6)

    assert result.is_grounded
    assert result.score >= 0.8
    assert len(result.unsupported_claims) == 0


def test_hallucinated_answer_fails() -> None:
    passages = [
        _make_chunk("To roll back a deployment, run kubectl rollout undo deployment/api-server.")
    ]
    # Answer hallucinates completely unmentioned AWS commands and Terraform concepts
    hallucinated_answer = (
        "According to our cloud infrastructure manual, you should execute terraform destroy "
        "and provision an Amazon ECS cluster with CloudWatch metrics alerts."
    )

    result = check_groundedness(hallucinated_answer, passages, min_score=0.6)

    assert not result.is_grounded
    assert result.score < 0.6
    assert len(result.unsupported_claims) > 0


def test_standard_refusal_is_always_grounded() -> None:
    passages = [_make_chunk("Some arbitrary document content.")]
    result = check_groundedness(NOT_FOUND_RESPONSE, passages, min_score=0.6)

    assert result.is_grounded
    assert result.score == 1.0


def test_empty_passages_with_factual_answer_fails() -> None:
    result = check_groundedness("The password is password123.", passages=[], min_score=0.6)

    assert not result.is_grounded
    assert result.score == 0.0
