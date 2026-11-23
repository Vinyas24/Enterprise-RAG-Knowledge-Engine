"""Unit tests for generation, prompt assembly, LLM providers, and citation mapping."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.generation.citations import extract_citations
from rag_engine.generation.deterministic import DeterministicLLMProvider
from rag_engine.generation.errors import LLMResponseError, LLMUnavailable
from rag_engine.generation.factory import build_llm, build_query_engine
from rag_engine.generation.openai import OpenAILLMProvider
from rag_engine.generation.prompts import (
    NOT_FOUND_RESPONSE,
    SYSTEM_PROMPT,
    assemble_prompt,
    format_passage,
)
from rag_engine.generation.service import QueryEngine
from rag_engine.retrieval.base import RetrievedChunk


def _make_chunk(
    index: int = 0,
    title: str = "Test Doc",
    content: str = "Retrieval augmented generation combines search and generation.",
    page: int | None = 1,
    section: str | None = "Overview",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title=title,
        chunk_index=index,
        content=content,
        score=0.92,
        access_level=AccessLevel.INTERNAL,
        page_start=page,
        page_end=page,
        section_path=section,
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_format_passage_with_locations() -> None:
    chunk1 = _make_chunk(page=3, section="Architecture")
    p1 = format_passage(1, chunk1)
    assert 'Passage [1] - Document: "Test Doc" (Page 3, Section: Architecture):' in p1
    assert chunk1.content in p1

    chunk2 = _make_chunk(page=None, section=None)
    object.__setattr__(chunk2, "page_start", 2)
    object.__setattr__(chunk2, "page_end", 4)
    p2 = format_passage(2, chunk2)
    assert 'Passage [2] - Document: "Test Doc" (Pages 2-4):' in p2


def test_assemble_prompt_empty_and_populated() -> None:
    sys_prompt, user_empty = assemble_prompt("What is RAG?", [])
    assert sys_prompt == SYSTEM_PROMPT
    assert "(No relevant passages found)" in user_empty

    chunk = _make_chunk()
    _, user_populated = assemble_prompt("What is RAG?", [chunk])
    assert "Passage [1]" in user_populated
    assert "Question: What is RAG?" in user_populated


# ---------------------------------------------------------------------------
# DeterministicLLMProvider
# ---------------------------------------------------------------------------


async def test_deterministic_llm_answers_from_context() -> None:
    llm = DeterministicLLMProvider()
    chunk = _make_chunk(content="The chunk overlap is 200 characters. Each chunk has metadata.")
    _, prompt = assemble_prompt("What is the chunk overlap?", [chunk])

    answer = await llm.generate(prompt)
    assert "The chunk overlap is 200 characters" in answer
    assert "[1]" in answer

    await llm.close()


async def test_deterministic_llm_not_found_on_irrelevant_question() -> None:
    llm = DeterministicLLMProvider()
    chunk = _make_chunk(content="The system uses PostgreSQL for metadata storage.")
    _, prompt = assemble_prompt("How many planets are in the solar system?", [chunk])

    answer = await llm.generate(prompt)
    assert answer == NOT_FOUND_RESPONSE


async def test_deterministic_llm_not_found_on_empty_context() -> None:
    llm = DeterministicLLMProvider()
    _, prompt = assemble_prompt("What is RAG?", [])
    answer = await llm.generate(prompt)
    assert answer == NOT_FOUND_RESPONSE


# ---------------------------------------------------------------------------
# OpenAILLMProvider
# ---------------------------------------------------------------------------


def test_openai_llm_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        OpenAILLMProvider(api_key="", model="gpt-4o-mini")


async def test_openai_llm_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "RAG improves accuracy [1]."}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    llm = OpenAILLMProvider(api_key="sk-test", model="gpt-4o-mini", transport=transport)
    res = await llm.generate("Question: What is RAG?", system_prompt="System instructions")
    assert res == "RAG improves accuracy [1]."
    await llm.close()


async def test_openai_llm_transient_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Error")

    transport = httpx.MockTransport(handler)
    llm = OpenAILLMProvider(api_key="sk-test", model="gpt-4o-mini", transport=transport)
    with pytest.raises(LLMUnavailable, match="500"):
        await llm.generate("prompt")
    await llm.close()


async def test_openai_llm_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    llm = OpenAILLMProvider(api_key="sk-test", model="gpt-4o-mini", transport=transport)
    with pytest.raises(LLMResponseError, match="missing choices"):
        await llm.generate("prompt")
    await llm.close()


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


def test_extract_citations_bracket_mapping() -> None:
    c1 = _make_chunk(index=0, title="Doc A", content="Content of chunk 0.")
    c2 = _make_chunk(index=1, title="Doc B", content="Content of chunk 1.")
    passages = [c1, c2]

    # Explicit [2] citation
    citations = extract_citations("According to the manual [2], setup is simple.", passages)
    assert len(citations) == 1
    assert citations[0].document_title == "Doc B"
    assert citations[0].chunk_index == 1

    # Multiple citations: [1] and [2]
    citations_multi = extract_citations("Facts from [1] and also [2].", passages)
    assert len(citations_multi) == 2
    assert [c.document_title for c in citations_multi] == ["Doc A", "Doc B"]

    # Unknown [99] ignored
    citations_oob = extract_citations("Mystery fact [99].", passages)
    assert len(citations_oob) == 0

    # Fallback to top passage if answering without explicit brackets
    citations_fallback = extract_citations("The answer is forty-two.", passages)
    assert len(citations_fallback) == 1
    assert citations_fallback[0].document_title == "Doc A"

    # No citations on cannot answer
    citations_none = extract_citations(NOT_FOUND_RESPONSE, passages)
    assert len(citations_none) == 0


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------


async def test_query_engine_weak_retrieval_returns_not_found() -> None:
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    llm = AsyncMock()
    settings = Settings()

    engine = QueryEngine(retriever=retriever, llm=llm, settings=settings)
    result = await engine.answer(tenant_id=uuid4(), question="Unanswerable question")

    assert result.answer == NOT_FOUND_RESPONSE
    assert result.citations == []
    assert result.weak_retrieval is True
    assert llm.generate.call_count == 0


async def test_query_engine_answers_with_citations() -> None:
    chunk = _make_chunk(title="Architecture Spec", content="Postgres stores metadata.")
    retriever = AsyncMock()
    retriever.retrieve.return_value = [chunk]

    llm = AsyncMock()
    llm.generate.return_value = "Postgres is used for metadata storage [1]."
    settings = Settings()

    engine = QueryEngine(retriever=retriever, llm=llm, settings=settings)
    result = await engine.answer(tenant_id=uuid4(), question="Where is metadata stored?")

    assert result.answer == "Postgres is used for metadata storage [1]."
    assert len(result.citations) == 1
    assert result.citations[0].document_title == "Architecture Spec"
    assert result.weak_retrieval is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_llm_factory() -> None:
    settings_det = Settings(llm_provider="deterministic")
    assert isinstance(build_llm(settings_det), DeterministicLLMProvider)

    settings_openai = Settings(llm_provider="openai", llm_api_key=SecretStr("sk-test"))
    assert isinstance(build_llm(settings_openai), OpenAILLMProvider)

    bad_settings = Settings()
    object.__setattr__(bad_settings, "llm_provider", "unknown")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_llm(bad_settings)


def test_build_query_engine_factory() -> None:
    settings = Settings()
    engine = build_query_engine(settings, retriever=AsyncMock(), llm=AsyncMock())
    assert isinstance(engine, QueryEngine)
