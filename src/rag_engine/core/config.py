"""Application configuration, loaded from environment variables.

All settings use the ``RAG_`` prefix and are read from the process environment
first, then from a local ``.env`` file (which is git-ignored). ``.env.example``
documents every key.

The settings object is immutable and cached: configuration is resolved once at
startup, so a running process cannot drift from its declared configuration.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Environment(StrEnum):
    """Deployment environment. Drives docs exposure and log defaults."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """Log rendering format."""

    JSON = "json"
    CONSOLE = "console"


class RetrievalMode(StrEnum):
    """Which retrieval paths feed the result list."""

    DENSE = "dense"
    HYBRID = "hybrid"


class RerankProvider(StrEnum):
    """How retrieved candidates are re-scored before the final cut."""

    OFF = "off"
    COHERE = "cohere"
    CROSS_ENCODER = "cross-encoder"
    # Deterministic BM25 over the merged pool. No model, no network: it exists
    # so offline evaluation and local development exercise the re-ranking path.
    LEXICAL = "lexical"


class Settings(BaseSettings):
    """Typed, validated, environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # --- application ------------------------------------------------------
    app_name: str = Field(default="rag-engine", min_length=1, max_length=64)
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: LogLevel = "INFO"
    # Structured JSON by default: logs are a machine interface, and the human
    # friendly console format is an explicit opt-in for local development.
    log_format: LogFormat = LogFormat.JSON

    # --- database ---------------------------------------------------------
    database_url: str = "postgresql+asyncpg://rag:rag_dev_password@localhost:5432/rag"
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=10, ge=0, le=50)
    database_pool_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    database_pool_recycle_seconds: int = Field(default=1800, ge=60, le=86400)
    database_connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    database_echo: bool = False

    # --- health checks ----------------------------------------------------
    # Bounds the readiness probe so a dead database can never hang the probe.
    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    # --- ingestion --------------------------------------------------------
    blob_store_path: Path = Path("data/uploads")
    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1024, le=1024**3)
    chunk_size_chars: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=4000)
    # Selected through the chunking factory, so alternative strategies can be
    # benchmarked later without changing the worker.
    chunk_strategy: Literal["recursive"] = "recursive"

    # --- ingestion worker -------------------------------------------------
    worker_poll_interval_seconds: float = Field(default=0.5, gt=0, le=60)
    worker_max_attempts: int = Field(default=3, ge=1, le=10)
    # Exponential backoff between retries: base * 2^(attempt-1), capped.
    worker_retry_base_seconds: float = Field(default=2.0, gt=0, le=300)
    worker_retry_max_seconds: float = Field(default=60.0, gt=0, le=3600)
    # A job stuck in "processing" longer than this is considered abandoned by a
    # dead worker and becomes claimable again.
    worker_visibility_timeout_seconds: float = Field(default=300.0, gt=0, le=86400)

    # --- embedding --------------------------------------------------------
    # "deterministic" needs no API key and is the default for local/test.
    # "openai" is the OpenAI-compatible provider for production.
    embedding_provider: Literal["deterministic", "openai"] = "deterministic"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=256, ge=8, le=4096)
    embedding_api_key: SecretStr | None = None
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    embedding_max_retries: int = Field(default=4, ge=0, le=20)
    embedding_retry_base_seconds: float = Field(default=0.5, gt=0, le=60)
    embedding_retry_max_seconds: float = Field(default=20.0, gt=0, le=300)

    # --- vector store -----------------------------------------------------
    # "memory" for local/test (no network), "pinecone" for production.
    vector_store: Literal["memory", "pinecone"] = "memory"
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "rag-engine"
    pinecone_metric: str = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_control_host: str | None = None
    pinecone_index_host: str | None = None
    pinecone_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    vector_upsert_batch_size: int = Field(default=100, ge=1, le=1000)
    vector_delete_batch_size: int = Field(default=1000, ge=1, le=10000)
    vector_list_page_size: int = Field(default=100, ge=1, le=10000)

    # --- retrieval --------------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # "dense" keeps Phase 4 behaviour; "hybrid" fuses dense and BM25 keyword
    # results. Hybrid is the default because keyword recall helps most on the
    # exact identifiers and error strings enterprise corpora are full of.
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    # Reciprocal-rank-fusion weights. They are ranks, not scores, so the scale
    # of a cosine similarity and a BM25 score never has to be reconciled.
    hybrid_dense_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    hybrid_keyword_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    # 60 is the value from the original RRF paper and dampens rank differences
    # without flattening the top of the list.
    hybrid_rrf_k: int = Field(default=60, ge=1, le=1000)
    # PostgreSQL selects this many keyword candidates before BM25 scoring.
    # Bounded so one very common term cannot make a query scan every chunk.
    retrieval_keyword_candidates: int = Field(default=200, ge=1, le=1000)
    # Resilience: graceful fallback if one of the hybrid paths fails or times out
    hybrid_fallback_enabled: bool = True
    dense_retrieval_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    keyword_retrieval_timeout_seconds: float = Field(default=10.0, gt=0, le=60)

    # --- re-ranking -------------------------------------------------------
    # "off" keeps Phase 6 behaviour. A configured re-ranker scores the merged
    # candidates and trims them to ``retrieval_top_k``.
    rerank_provider: RerankProvider = RerankProvider.OFF
    # Provider default applies when unset, because the same field carries a
    # Cohere model id and a local sentence-transformers model path.
    rerank_model: str | None = None
    # The merged pool the re-ranker sees. Bounded so a single query cannot send
    # an unbounded prompt to a hosted provider.
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    # A deadline, not a retry budget: past it the caller gets the un-reranked
    # order rather than a slow answer.
    rerank_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    rerank_api_key: SecretStr | None = None
    rerank_api_base: str = "https://api.cohere.com/v2"
    # Cross-encoder only: pairs per forward pass, and the device string the
    # model is loaded on (unset lets sentence-transformers choose).
    rerank_batch_size: int = Field(default=32, ge=1, le=256)
    rerank_device: str | None = None

    # --- generation / LLM -------------------------------------------------
    # "deterministic" for local/tests without external API keys; "openai" for production.
    llm_provider: Literal["deterministic", "openai"] = "deterministic"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_api_base: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1, le=8192)

    # --- guardrails -------------------------------------------------------
    guardrails_enabled: bool = True
    guardrails_max_query_length: int = Field(default=2000, ge=1, le=10000)
    guardrails_min_query_length: int = Field(default=1, ge=1, le=100)
    guardrails_mask_pii: bool = True
    guardrails_detect_prompt_injection: bool = True
    guardrails_check_groundedness: bool = True
    guardrails_min_groundedness_score: float = Field(default=0.6, ge=0.0, le=1.0)
    guardrails_validate_citations: bool = True
    guardrails_rate_limit_per_minute: int = Field(default=120, ge=1, le=10000)
    guardrails_audit_log_enabled: bool = True

    # --- caching (Phase 9) ------------------------------------------------
    cache_enabled: bool = True
    cache_embedding_enabled: bool = True
    cache_embedding_max_size: int = Field(default=10000, ge=100, le=100000)
    cache_embedding_ttl_seconds: float = Field(default=3600.0, gt=0, le=86400)
    cache_retrieval_enabled: bool = True
    cache_retrieval_max_size: int = Field(default=2000, ge=50, le=50000)
    cache_retrieval_ttl_seconds: float = Field(default=300.0, gt=0, le=86400)

    # --- observability & polish (Phase 10) ---------------------------------
    auth_enabled: bool = False
    api_key: str | None = None
    admin_api_key: str | None = None
    metrics_enabled: bool = True
    server_timing_enabled: bool = True

    @field_validator(
        "rerank_api_key",
        "rerank_model",
        "rerank_device",
        "api_key",
        "admin_api_key",
        mode="before",
    )
    @classmethod
    def _blank_rerank_values_are_unset(cls, value: object) -> object:
        """Treat ``RAG_X=`` from Compose as unset rather than an empty string.

        ``device=""`` would otherwise be passed to the model loader, and an
        empty secret would look configured to the key check.
        """
        return None if value == "" else value

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        """Fail fast on a sync DSN.

        ``postgresql://`` silently selects psycopg2, which cannot be used by
        the async engine. Catching it here turns a runtime crash into a clear
        startup error.
        """
        try:
            url = make_url(value)
        except Exception as exc:
            raise ValueError(f"database_url is not a valid URL: {exc}") from exc
        if url.drivername != "postgresql+asyncpg":
            raise ValueError(
                "database_url must use the asyncpg driver, "
                f"got {url.drivername!r} (expected 'postgresql+asyncpg')"
            )
        return value

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        return self

    @model_validator(mode="after")
    def _validate_hybrid_weights(self) -> Settings:
        """Reject a fusion configuration where every path contributes nothing.

        Checked regardless of ``retrieval_mode`` so flipping the mode later
        cannot surface a configuration that silently returns no results.
        """
        if self.hybrid_dense_weight == 0.0 and self.hybrid_keyword_weight == 0.0:
            raise ValueError("hybrid_dense_weight and hybrid_keyword_weight cannot both be zero")
        return self

    @model_validator(mode="after")
    def _validate_rerank_key(self) -> Settings:
        """Require an API key when the hosted re-ranker is selected."""
        if self.rerank_provider is RerankProvider.COHERE and not self.rerank_api_key:
            raise ValueError("rerank_api_key is required when rerank_provider is 'cohere'")
        return self

    @model_validator(mode="after")
    def _validate_embedding_key(self) -> Settings:
        """Require an API key when the OpenAI provider is selected."""
        if self.embedding_provider == "openai" and not self.embedding_api_key:
            raise ValueError("embedding_api_key is required when embedding_provider is 'openai'")
        return self

    @model_validator(mode="after")
    def _validate_pinecone_key(self) -> Settings:
        """Require an API key when Pinecone is selected."""
        if self.vector_store == "pinecone" and not self.pinecone_api_key:
            raise ValueError("pinecone_api_key is required when vector_store is 'pinecone'")
        return self

    @model_validator(mode="after")
    def _validate_llm_key(self) -> Settings:
        """Require an API key when the OpenAI LLM provider is selected."""
        if self.llm_provider == "openai" and not self.llm_api_key:
            raise ValueError("llm_api_key is required when llm_provider is 'openai'")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @property
    def docs_enabled(self) -> bool:
        """Interactive API docs are disabled in production by default."""
        return not self.is_production

    @property
    def redacted_database_url(self) -> str:
        """DSN safe to log: credentials are masked."""
        return make_url(self.database_url).render_as_string(hide_password=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
