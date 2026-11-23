"""Generation contracts and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation mapping an answer back to its originating document chunk."""

    document_id: UUID
    document_title: str
    chunk_id: UUID
    chunk_index: int
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The synthesized answer along with verified citations."""

    answer: str
    citations: list[Citation]
    weak_retrieval: bool


@runtime_checkable
class LLMProvider(Protocol):
    """Pluggable text generation interface."""

    name: str

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from a prompt."""
        ...

    async def close(self) -> None:
        """Release any client resources. Idempotent."""
        ...
