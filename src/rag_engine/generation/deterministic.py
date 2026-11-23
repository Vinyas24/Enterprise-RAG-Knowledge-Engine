"""Deterministic, hermetic LLM provider for tests and local development."""

from __future__ import annotations

import re

from .base import LLMProvider
from .prompts import NOT_FOUND_RESPONSE

_WORD_RE = re.compile(r"\b[a-zA-Z0-9_]{3,}\b")
_PASSAGE_RE = re.compile(
    r"Passage \[(\d+)\][^\n]*:\n(.*?)(?=\n\nPassage \[\d+\]|\n\nQuestion:|$)",
    re.DOTALL,
)
_STOP_WORDS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "does",
    "is",
    "are",
    "the",
    "and",
    "for",
    "with",
    "about",
    "many",
    "much",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "name",
    "was",
    "were",
    "been",
}


class DeterministicLLMProvider(LLMProvider):
    """Answers strictly based on passage keyword overlap, citing the source passage."""

    name = "deterministic"

    async def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if "(No relevant passages found)" in prompt:
            return NOT_FOUND_RESPONSE

        # Extract question
        q_match = re.search(r"Question:\s*(.*?)(?:\n\nAnswer:|$)", prompt, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        q_words = set(_WORD_RE.findall(question.lower())) - _STOP_WORDS

        # Extract passages
        passages = _PASSAGE_RE.findall(prompt)
        if not passages or not q_words:
            return NOT_FOUND_RESPONSE

        min_required = 1 if len(q_words) <= 2 else 2

        best_passage_num: int | None = None
        best_sentences: list[str] = []
        best_overlap = 0

        for num_str, content in passages:
            num = int(num_str)
            sentences = [
                s.strip()
                for s in re.split(
                    r"(?:(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!vs\.)(?<=[.!?])\s+|\n+)",
                    content.strip(),
                )
                if s.strip()
            ]
            passage_sentence_overlaps = []
            for sentence in sentences:
                s_words = set(_WORD_RE.findall(sentence.lower()))
                overlap = len(q_words & s_words)
                passage_sentence_overlaps.append((overlap, sentence))

            max_in_passage = max((o for o, _ in passage_sentence_overlaps), default=0)
            if max_in_passage > best_overlap:
                best_overlap = max_in_passage
                best_passage_num = num
                best_sentences = [
                    s
                    for o, s in passage_sentence_overlaps
                    if o == max_in_passage and o >= min_required
                ]

        if best_sentences and best_passage_num is not None and best_overlap >= min_required:
            joined = " ".join(best_sentences)
            return f"According to the provided documents, {joined} [{best_passage_num}]"

        return NOT_FOUND_RESPONSE

    async def close(self) -> None:
        return None
