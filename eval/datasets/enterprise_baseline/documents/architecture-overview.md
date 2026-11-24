# Architecture Overview

## Services

The platform has a FastAPI API, a separate ingestion worker, a PostgreSQL database and a Pinecone vector index. The API never parses documents; it writes a job and returns immediately. The worker scales independently because parsing and embedding are CPU and network heavy.

## Asynchronous processing

Ingestion jobs are stored in the PostgreSQL jobs table and claimed with `SELECT ... FOR UPDATE SKIP LOCKED`. This gives at-least-once delivery without a separate broker. A visibility timeout reclaims jobs from a crashed worker, and deterministic chunk ids make retries idempotent.

## Data stores

PostgreSQL stores tenant metadata, document rows, chunk text and provenance. Pinecone stores embeddings in one namespace per tenant. Redis caches short-lived query results and rate limit counters in the request path; it is never the source of truth.

## Observability

Every request carries a validated request id that appears in structured logs and error envelopes. Readiness checks PostgreSQL, liveness does not. Dashboards track request latency, ingestion queue depth and vector store errors.
