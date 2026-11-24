"""Evaluation dataset schema and loader.

A dataset is deliberately data, not code: documents plus questions with the
documents that should answer them. Keeping it in JSON means a change to the
corpus or the labels is reviewable in a diff, and the fingerprint recorded in a
baseline makes "the dataset changed" impossible to miss.

Document bodies can live inline (``content``) or in a sibling file (``path``).
The committed dataset uses files so the prose stays readable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_engine.core.enums import AccessLevel


class EvalDocument(BaseModel):
    """One corpus document, identified by a stable human-readable key."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1, max_length=512)
    filename: str = Field(default="document.md", min_length=1, max_length=255)
    access_level: AccessLevel = AccessLevel.INTERNAL
    content: str = Field(min_length=1)


class EvalQuestion(BaseModel):
    """A question and the document(s) that contain its answer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    relevant_documents: list[str] = Field(min_length=1)
    clearance: AccessLevel = AccessLevel.INTERNAL


class EvalDataset(BaseModel):
    """A complete, self-contained retrieval benchmark."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    version: int = Field(default=1, ge=1)
    description: str = ""
    documents: list[EvalDocument] = Field(min_length=1)
    questions: list[EvalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_references(self) -> EvalDataset:
        keys = [document.key for document in self.documents]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate document keys: {', '.join(duplicates)}")

        question_ids = [question.id for question in self.questions]
        duplicate_ids = sorted({qid for qid in question_ids if question_ids.count(qid) > 1})
        if duplicate_ids:
            raise ValueError(f"duplicate question ids: {', '.join(duplicate_ids)}")

        known = set(keys)
        unknown = sorted(
            {
                document_key
                for question in self.questions
                for document_key in question.relevant_documents
                if document_key not in known
            }
        )
        if unknown:
            raise ValueError(f"questions reference unknown documents: {', '.join(unknown)}")
        return self


def load_dataset(path: Path) -> EvalDataset:
    """Load a dataset manifest, resolving ``path`` entries relative to it."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    documents = raw.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"{path} must contain a 'documents' list")

    base = path.parent
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"{path}: document {index} must be an object")
        reference = document.pop("path", None)
        if reference is None:
            continue
        if "content" in document:
            raise ValueError(
                f"{path}: document {document.get('key', index)!r} has both 'path' and 'content'"
            )
        if not isinstance(reference, str):
            raise ValueError(f"{path}: document {index} has a non-string 'path'")
        document_path = (base / reference).resolve()
        try:
            document["content"] = document_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{path}: cannot read {document_path}: {exc}") from exc

    return EvalDataset.model_validate(raw)


def dataset_fingerprint(dataset: EvalDataset) -> str:
    """Stable hash of the corpus and labels, used to detect stale baselines."""
    payload = dataset.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
