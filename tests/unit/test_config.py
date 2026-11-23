"""Configuration behaviour: env overrides, validation and derived properties."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_engine.core.config import Environment, LogFormat, Settings, get_settings


def test_defaults_are_local_and_structured() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment is Environment.LOCAL
    assert settings.log_format is LogFormat.JSON
    assert settings.log_level == "INFO"
    assert settings.database_pool_size == 5
    assert settings.docs_enabled is True


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENVIRONMENT", "staging")
    monkeypatch.setenv("RAG_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("RAG_LOG_FORMAT", "console")
    monkeypatch.setenv("RAG_DATABASE_POOL_SIZE", "12")

    settings = Settings(_env_file=None)

    assert settings.environment is Environment.STAGING
    assert settings.log_level == "WARNING"
    assert settings.log_format is LogFormat.CONSOLE
    assert settings.database_pool_size == 12


def test_sync_database_driver_is_rejected() -> None:
    with pytest.raises(ValidationError, match="asyncpg"):
        Settings(
            database_url="postgresql://rag:secret@localhost:5432/rag",
            _env_file=None,
        )


def test_malformed_database_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not a valid URL"):
        Settings(database_url="not a url", _env_file=None)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("TRACE", "Input should be"),
        ("info", "Input should be"),
        ("", "Input should be"),
    ],
)
def test_invalid_log_level_is_rejected(value: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings(log_level=value, _env_file=None)


def test_pool_size_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(database_pool_size=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(database_pool_size=51, _env_file=None)


def test_health_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(health_check_timeout_seconds=0, _env_file=None)


def test_production_disables_docs() -> None:
    settings = Settings(environment=Environment.PRODUCTION, _env_file=None)

    assert settings.is_production is True
    assert settings.docs_enabled is False


def test_database_url_is_redacted_for_logging() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://rag:supersecret@db.internal:5432/rag",
        _env_file=None,
    )

    redacted = settings.redacted_database_url

    assert "supersecret" not in redacted
    assert "db.internal" in redacted
    assert redacted.startswith("postgresql+asyncpg://")


def test_settings_are_immutable() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_chars"):
        Settings(chunk_size_chars=500, chunk_overlap_chars=500, _env_file=None)


def test_only_known_chunking_strategies_are_accepted() -> None:
    with pytest.raises(ValidationError, match="chunk_strategy"):
        Settings(chunk_strategy="semantic", _env_file=None)


def test_ingestion_defaults_are_documented_and_bounded() -> None:
    settings = Settings(_env_file=None)

    assert settings.upload_max_bytes == 25 * 1024 * 1024
    assert settings.chunk_size_chars == 1200
    assert settings.chunk_overlap_chars == 200
    assert settings.chunk_strategy == "recursive"
    assert settings.worker_max_attempts == 3
    assert settings.worker_visibility_timeout_seconds > settings.worker_retry_max_seconds


def test_upload_size_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(upload_max_bytes=1, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(upload_max_bytes=1024**3 + 1, _env_file=None)


def test_embedding_and_vectorstore_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "deterministic"
    assert settings.embedding_dimensions == 256
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.vector_store == "memory"
    assert settings.pinecone_index_name == "rag-engine"
    assert settings.pinecone_metric == "cosine"


def test_openai_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="embedding_api_key is required"):
        Settings(embedding_provider="openai", _env_file=None)


def test_pinecone_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="pinecone_api_key is required"):
        Settings(vector_store="pinecone", _env_file=None)
