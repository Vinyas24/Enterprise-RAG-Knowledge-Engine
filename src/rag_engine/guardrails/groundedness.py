"""Output guardrail: Groundedness check of answers against retrieved context.

Ensures generated statements are factually rooted in the provided passages
rather than hallucinated from model parametric memory.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_engine.generation.prompts import NOT_FOUND_RESPONSE
from rag_engine.retrieval.base import RetrievedChunk
from rag_engine.retrieval.keyword import tokenize

from .base import GroundednessResult

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "their",
    "them",
    "we",
    "our",
    "us",
    "you",
    "your",
}


def _clean_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in _STOPWORDS and not token.isdigit()
    }


def check_groundedness(
    answer: str,
    passages: Sequence[RetrievedChunk],
    min_score: float = 0.6,
) -> GroundednessResult:
    """Evaluate whether the claims in ``answer`` are grounded in ``passages``."""
    clean_answer = answer.strip()
    if not clean_answer:
        return GroundednessResult(is_grounded=True, score=1.0)

    # Standard refusals are inherently grounded: they state inability to answer
    if clean_answer == NOT_FOUND_RESPONSE or clean_answer.startswith("I cannot answer"):
        return GroundednessResult(
            is_grounded=True,
            score=1.0,
            supported_claims=[clean_answer],
            unsupported_claims=[],
        )

    if not passages:
        return GroundednessResult(
            is_grounded=False,
            score=0.0,
            supported_claims=[],
            unsupported_claims=[clean_answer],
        )

    # Aggregate passage tokens
    passage_tokens = set()
    for chunk in passages:
        passage_tokens.update(_clean_tokens(chunk.content))

    # Split answer into sentences / claims
    raw_sentences = _SENTENCE_SPLIT_RE.split(clean_answer)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return GroundednessResult(is_grounded=True, score=1.0)

    supported: list[str] = []
    unsupported: list[str] = []

    for sentence in sentences:
        sentence_tokens = _clean_tokens(sentence)
        if not sentence_tokens:
            # Punctuation/fillers without distinct content words count as supported
            supported.append(sentence)
            continue

        overlap = sentence_tokens.intersection(passage_tokens)
        overlap_ratio = len(overlap) / len(sentence_tokens)

        # A sentence is supported if at least 40% of its content words are verified in context
        if overlap_ratio >= 0.40:
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    score = len(supported) / len(sentences)
    is_grounded = score >= min_score

    return GroundednessResult(
        is_grounded=is_grounded,
        score=score,
        supported_claims=supported,
        unsupported_claims=unsupported,
    )
