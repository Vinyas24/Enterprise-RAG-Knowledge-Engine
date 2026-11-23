"""Prompt construction and strict contextual answering templates."""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.retrieval.base import RetrievedChunk

SYSTEM_PROMPT = """You are a strictly factual assistant for an enterprise knowledge engine.
Your duty is to answer the user's question using ONLY the provided context passages.

Strict Instructions:
1. Answer using ONLY information directly provided in the context passages. \
Do NOT extrapolate, hypothesize, or use outside knowledge.
2. If the context does not contain enough information to answer the question, respond with:
"I cannot answer this question based on the provided documents."
3. Every factual claim in your answer MUST cite its source using bracketed numbers \
corresponding to the passage, e.g. [1] or [2].
"""

NOT_FOUND_RESPONSE = "I cannot answer this question based on the provided documents."


def format_passage(index: int, chunk: RetrievedChunk) -> str:
    """Format one passage with source header."""
    location_parts = []
    if chunk.page_start is not None:
        if chunk.page_end is not None and chunk.page_end != chunk.page_start:
            location_parts.append(f"Pages {chunk.page_start}-{chunk.page_end}")
        else:
            location_parts.append(f"Page {chunk.page_start}")
    if chunk.section_path:
        location_parts.append(f"Section: {chunk.section_path}")

    loc_str = f" ({', '.join(location_parts)})" if location_parts else ""
    title = chunk.document_title
    return f'Passage [{index}] - Document: "{title}"{loc_str}:\n{chunk.content.strip()}'


def assemble_prompt(question: str, passages: Sequence[RetrievedChunk]) -> tuple[str, str]:
    """Assemble system and user prompts with numbered context passages."""
    if not passages:
        user_prompt = (
            f"Context passages:\n(No relevant passages found)\n\n"
            f"Question: {question.strip()}\n\n"
            f"Answer:"
        )
        return SYSTEM_PROMPT, user_prompt

    formatted_passages = "\n\n".join(
        format_passage(i + 1, chunk) for i, chunk in enumerate(passages)
    )

    user_prompt = (
        f"Context passages:\n{formatted_passages}\n\nQuestion: {question.strip()}\n\nAnswer:"
    )
    return SYSTEM_PROMPT, user_prompt
