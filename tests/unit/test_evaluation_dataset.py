"""Dataset loading, validation and fingerprinting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_engine.evaluation.dataset import (
    EvalDataset,
    dataset_fingerprint,
    load_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_DATASET = PROJECT_ROOT / "eval" / "datasets" / "enterprise_baseline.json"


def _manifest(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "tiny",
        "version": 1,
        "documents": [
            {
                "key": "alpha",
                "title": "Alpha",
                "filename": "alpha.md",
                "content": "# Alpha\n\nAlpha content.",
            }
        ],
        "questions": [
            {
                "id": "q1",
                "question": "What is alpha?",
                "relevant_documents": ["alpha"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_dataset_accepts_inline_content(tmp_path: Path) -> None:
    dataset = load_dataset(_write(tmp_path / "dataset.json", _manifest()))

    assert dataset.name == "tiny"
    assert dataset.documents[0].content == "# Alpha\n\nAlpha content."
    assert dataset.questions[0].relevant_documents == ["alpha"]


def test_load_dataset_resolves_paths_relative_to_the_manifest(tmp_path: Path) -> None:
    documents = tmp_path / "corpus"
    documents.mkdir()
    (documents / "alpha.md").write_text("# Alpha\n\nFrom a file.", encoding="utf-8")
    payload = _manifest()
    payload["documents"][0].pop("content")
    payload["documents"][0]["path"] = "corpus/alpha.md"

    dataset = load_dataset(_write(tmp_path / "dataset.json", payload))

    assert dataset.documents[0].content == "# Alpha\n\nFrom a file."


def test_load_dataset_rejects_both_path_and_content(tmp_path: Path) -> None:
    payload = _manifest()
    payload["documents"][0]["path"] = "alpha.md"

    with pytest.raises(ValueError, match="both 'path' and 'content'"):
        load_dataset(_write(tmp_path / "dataset.json", payload))


def test_load_dataset_reports_a_missing_document(tmp_path: Path) -> None:
    payload = _manifest()
    payload["documents"][0].pop("content")
    payload["documents"][0]["path"] = "missing.md"

    with pytest.raises(ValueError, match="cannot read"):
        load_dataset(_write(tmp_path / "dataset.json", payload))


def test_load_dataset_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_dataset(path)


def test_dataset_rejects_duplicate_document_keys() -> None:
    payload = _manifest()
    payload["documents"].append(dict(payload["documents"][0]))

    with pytest.raises(ValidationError, match="duplicate document keys"):
        EvalDataset.model_validate(payload)


def test_dataset_rejects_duplicate_question_ids() -> None:
    payload = _manifest()
    payload["questions"].append(dict(payload["questions"][0]))

    with pytest.raises(ValidationError, match="duplicate question ids"):
        EvalDataset.model_validate(payload)


def test_dataset_rejects_unknown_document_references() -> None:
    payload = _manifest()
    payload["questions"][0]["relevant_documents"] = ["ghost"]

    with pytest.raises(ValidationError, match="unknown documents: ghost"):
        EvalDataset.model_validate(payload)


def test_dataset_rejects_unknown_fields() -> None:
    payload = _manifest()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        EvalDataset.model_validate(payload)


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    dataset = EvalDataset.model_validate(_manifest())
    same = EvalDataset.model_validate(_manifest())
    changed = dataset.model_copy(update={"description": "a different description"})

    assert dataset_fingerprint(dataset) == dataset_fingerprint(same)
    assert dataset_fingerprint(dataset) != dataset_fingerprint(changed)


def test_committed_dataset_meets_the_phase_five_shape() -> None:
    dataset = load_dataset(COMMITTED_DATASET)

    assert 30 <= len(dataset.questions) <= 50
    assert len(dataset.documents) >= 5
    assert all(document.content.strip() for document in dataset.documents)
    assert all(question.relevant_documents for question in dataset.questions)

    referenced = {key for question in dataset.questions for key in question.relevant_documents}
    assert referenced == {document.key for document in dataset.documents}
