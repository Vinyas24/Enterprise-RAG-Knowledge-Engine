"""Query and answer generation service."""

from __future__ import annotations

import logging
from uuid import UUID

from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.retrieval.base import Retriever

from .base import GenerationResult, LLMProvider
from .citations import extract_citations
from .prompts import NOT_FOUND_RESPONSE, assemble_prompt

logger = logging.getLogger(__name__)


class QueryEngine:
    """Coordinates retrieval, prompt assembly, LLM generation, and citation mapping."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    async def answer(
        self,
        *,
        tenant_id: UUID,
        question: str,
        top_k: int | None = None,
        clearance: AccessLevel = AccessLevel.INTERNAL,
    ) -> GenerationResult:
        resolved_top_k = top_k or self._settings.retrieval_top_k
        clean_question = question.strip()

        # 1. Retrieve candidate chunks
        passages = await self._retriever.retrieve(
            tenant_id=tenant_id,
            query=clean_question,
            top_k=resolved_top_k,
            clearance=clearance,
            min_score=self._settings.retrieval_min_score,
        )

        # 2. Check for weak retrieval
        if not passages:
            logger.info("generation.weak_retrieval", extra={"tenant_id": str(tenant_id)})
            return GenerationResult(
                answer=NOT_FOUND_RESPONSE,
                citations=[],
                weak_retrieval=True,
            )

        # 3. Assemble strict contextual prompt
        system_prompt, user_prompt = assemble_prompt(clean_question, passages)

        # 4. Generate answer
        raw_answer = await self._llm.generate(user_prompt, system_prompt=system_prompt)

        # 5. Check if LLM determined the context does not answer the question
        if raw_answer.strip() == NOT_FOUND_RESPONSE or raw_answer.startswith("I cannot answer"):
            return GenerationResult(
                answer=NOT_FOUND_RESPONSE,
                citations=[],
                weak_retrieval=True,
            )

        # 6. Extract and verify citations
        citations = extract_citations(raw_answer, passages)

        return GenerationResult(
            answer=raw_answer,
            citations=citations,
            weak_retrieval=False,
        )
