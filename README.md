# RAG Knowledge Engine

Enterprise-grade retrieval-augmented generation backend: ingest an
organization's documents, search them with hybrid retrieval, and answer
questions with grounded, cited responses — while enforcing tenant and access
control at retrieval time.

> **Status: Phase 2 — traceable chunking.**
> The service skeleton (API, configuration, logging, database connectivity,
> tests, CI, Compose stack) is in place, and documents can be uploaded, parsed,
> chunked and indexed asynchronously with boundary-aware chunks, exact source
> offsets and per-chunk access metadata. Retrieval, ranking, generation,
> guardrails and evaluation land in later phases.

---

## Quick start

### Docker Compose (recommended)

```bash
docker compose up --build
```

| Endpoint | Purpose |
| --- | --- |
| `http://localhost:8000/` | Service metadata |
| `http://localhost:8000/health/live` | Liveness: process is up (no dependencies) |
| `http://localhost:8000/health/ready` | Readiness: PostgreSQL is reachable |
| `http://localhost:8000/documents` | Upload (`POST`) and list (`GET`) documents |
| `http://localhost:8000/documents/{id}` | Document status, metadata, latest job |
| `http://localhost:8000/ingestion-jobs/{id}` | Ingestion job status and errors |
| `http://localhost:8000/docs` | OpenAPI UI (disabled in production) |

`docker compose` reads `.env` if present; without one it falls back to the
development defaults in `docker-compose.yml` and `.env.example`. If port 5432
is already taken on your machine, run with `POSTGRES_PORT=55432 docker compose
up --build` (the API always reaches PostgreSQL over the Compose network, so no
other setting changes).

```bash
docker compose ps                       # stack status (api, worker, postgres)
docker compose logs -f api worker       # structured logs
docker compose exec api pytest          # tests inside the container
docker compose exec api rag-engine list-tenants
docker compose down                     # stop (keeps the DB volume)
```

The `migrate` service applies Alembic migrations once; the API and worker wait
for it to finish, so no process ever serves against a stale schema.

### Local Python

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d postgres
make migrate                            # create the schema
rag-engine create-tenant --slug acme --name "Acme Corp"
uvicorn rag_engine.main:app --reload    # terminal 1: API
python -m rag_engine.ingestion.worker   # terminal 2: background ingestion
make check                              # lint + typecheck + full tests
```

---

## Repository layout

```
src/rag_engine/
├── api/            HTTP layer: app factory, routers, middleware, errors
│   └── routers/    health, metadata, documents, ingestion jobs
├── core/           configuration, logging, request context, database, models
├── ingestion/      upload → parse → chunk → persist, worker, retry, CLI
│   └── parsers/    PDF, DOCX, Markdown, TXT behind one interface
├── retrieval/      dense + BM25 retrieval, candidate fusion  (Phase 3+)
├── ranking/        cross-encoder / Cohere re-ranking         (Phase 4+)
├── generation/     prompt assembly, LLM calls, citations     (Phase 4+)
└── guardrails/     input checks, grounding, access control   (Phase 5+)
tests/
├── unit/           hermetic: no database, no network
└── integration/    real PostgreSQL via RAG_TEST_DATABASE_URL
```

**Import direction is one-way:** `api` → domain slices → `core`. Domain slices
never import from `api`, so the pipeline stays usable outside HTTP (workers,
CLI, evaluation harness) and is testable without a server.

---

## Configuration

Every setting is an environment variable with the `RAG_` prefix. `.env.example`
is the authoritative list; `.env` is git-ignored.

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_ENVIRONMENT` | `local` | `local`/`test`/`staging`/`production` |
| `RAG_LOG_LEVEL` | `INFO` | `DEBUG`…`CRITICAL` |
| `RAG_LOG_FORMAT` | `json` | `json` for machines, `console` for humans |
| `RAG_DATABASE_URL` | dev DSN | **Must** use `postgresql+asyncpg://` |
| `RAG_DATABASE_POOL_SIZE` | `5` | Bounded 1–50 |
| `RAG_HEALTH_CHECK_TIMEOUT_SECONDS` | `2` | Bounds the readiness probe |
| `RAG_BLOB_STORE_PATH` | `data/uploads` | Where raw bytes are stored |
| `RAG_UPLOAD_MAX_BYTES` | `26214400` (25 MiB) | Hard upload ceiling |
| `RAG_CHUNK_SIZE_CHARS` | `1200` | Maximum chunk length |
| `RAG_CHUNK_OVERLAP_CHARS` | `200` | Context shared by adjacent chunks |
| `RAG_CHUNK_STRATEGY` | `recursive` | Chunking strategy selected by the factory |
| `RAG_WORKER_MAX_ATTEMPTS` | `3` | Attempts before a job is terminally failed |
| `RAG_WORKER_RETRY_BASE_SECONDS` | `2` | Exponential backoff base |
| `RAG_WORKER_RETRY_MAX_SECONDS` | `60` | Backoff cap |
| `RAG_WORKER_VISIBILITY_TIMEOUT_SECONDS` | `300` | Reclaim window for dead workers |

Invalid configuration fails at startup with a precise message rather than at
first use. A plain `postgresql://` DSN is rejected because it silently selects a
synchronous driver that the async engine cannot use.

---

## Ingestion (Phase 1)

Tenants are created administratively (there is no signup yet), then every data
request carries the tenant slug in `X-Tenant-ID`:

```bash
docker compose exec api rag-engine create-tenant --slug acme --name "Acme Corp"

# 202 Accepted as soon as the document and its job are committed
curl -X POST http://localhost:8000/documents \
  -H 'X-Tenant-ID: acme' \
  -F 'file=@handbook.pdf' \
  -F 'access_level=internal'

# -> {"document_id": "...", "job_id": "...", "status": "queued"}

curl -H 'X-Tenant-ID: acme' http://localhost:8000/ingestion-jobs/<job_id>
curl -H 'X-Tenant-ID: acme' http://localhost:8000/documents/<document_id>
```

**Lifecycle.** A document moves `pending → processing → indexed` (or `failed`);
a job moves `queued → processing → done` (or `failed`). Both are queryable
immediately after upload. Parsing happens in a separate worker service, so an
upload never waits on a 200-page PDF.

**Queue without Redis.** `ingestion_jobs` is the queue. Workers claim a row with
`SELECT ... FOR UPDATE SKIP LOCKED`, which lets several workers run safely and
makes a crashed worker's job claimable again after the visibility timeout. The
retry delay is exponential (`base * 2^(attempt-1)`, capped); after
`max_attempts` the job and document are terminally `failed`.

**Deduplication is content-based and per tenant.** SHA-256 of the raw bytes is
unique per `(tenant_id, content_hash)`. A duplicate upload returns `409` with
`details.document_id` instead of creating a second document. The unique
constraint — not the pre-check — is the source of truth, so concurrent uploads
of the same file cannot both win. Identical bytes uploaded by a different tenant
are a separate document: existence itself is tenant-private.

**Parsing and chunking.** PDF, DOCX, Markdown and TXT share one parser interface.
PDF blocks carry page numbers; DOCX and Markdown blocks carry heading paths;
plain text carries neither. Chunking is recursive and boundary-aware (detailed
in Phase 2 below), keeps overlap, and never lets a chunk cross a page or heading
boundary. A chunk is therefore always traceable to its source location, which is
what citations will need once retrieval and generation land.

**Storage and errors.** Raw bytes live behind a `BlobStore` interface, stored
content-addressed under `<tenant>/<ab>/<sha256><ext>` on a shared Compose
volume. Every error, including ingestion failures, uses one envelope:
`{"error": {"code", "message", "request_id", "details"}}`. Validation happens
before the body is buffered where possible: wrong extension (`415`), oversized
(`413`), empty (`422`), and content that does not match its extension (`415`).

### Phase 1 trade-offs

- PostgreSQL as a queue is enough at this scale and removes an operational
  dependency. It is not a substitute for a dedicated broker under very high
  enqueue rates.
- Raw bytes are stored before database rows. An orphaned blob is harmless
  (content-addressed and reusable); a row pointing at missing bytes is not.
- Scanned PDFs have no text layer and are rejected with an explicit error. OCR
  is deliberately out of scope rather than silently producing empty documents.
- The worker is at-least-once. Re-processing deletes and rewrites a document's
  chunks in one transaction, so retries are idempotent from the reader's view.

---

## Chunking (Phase 2)

Goal: documents become high-quality, traceable chunks. Retrieval quality starts
here; a fixed-width window cuts ideas in half, and no downstream model can
repair that.

**Strategy interface.** `ChunkingStrategy` is a Protocol with `name` and
`chunk(ParsedDocument) -> list[ChunkDraft]`. `RecursiveChunker` is the first
implementation, selected by `build_chunker(settings)` from
`RAG_CHUNK_STRATEGY`. A second strategy can be benchmarked without changing the
worker or the persistence layer.

**Boundary hierarchy.** The splitter tries, in order:

1. blank line (paragraph),
2. single newline (list item, table row, heading),
3. sentence end (abbreviation-, initial-, acronym- and decimal-aware),
4. clause punctuation (`;`, `,`),
5. word,
6. hard character cut, only when no boundary exists.

A table row or sentence is never cut while a coarser boundary is available.
Overlap is also taken from a sentence or word boundary, so a chunk never begins
with the tail of a word. If no boundary exists (one enormous URL or base64
blob), a hard cut is the only option, and that case is tested.

**Segmentation by provenance.** Blocks are grouped by `(page, section_path)`
before splitting, so a chunk never mixes two pages or two headings. PDF chunks
record `page_start`/`page_end`; Markdown and DOCX chunks record the heading path.

**Offsets are the citation contract.** Every draft carries `start_char` and
`end_char` into the parser's normalised text (non-empty blocks joined by a blank
line). Slicing that text at the reported span reproduces the chunk exactly,
including overlap. Unit tests assert the invariant directly; integration tests
verify it end-to-end through the API.

**Chunk metadata.** Each row stores the tenant and document ids, position
(`chunk_index`), content and length, page or section provenance, the character
span, and the document's access level. `access_level` is snapshotted at
ingestion time so retrieval can enforce clearance without joining `documents`;
the `(tenant_id, access_level)` index matches that filter exactly.

**Inspection.** Chunks are queryable in reading order:

```bash
curl -H 'X-Tenant-ID: acme' \
  'http://localhost:8000/documents/<document_id>/chunks?limit=50&offset=0'
```

```json
{
  "items": [{
    "document_id": "…",
    "tenant_id": "…",
    "chunk_index": 0,
    "content": "Deployment\n\nUse containers.",
    "char_count": 27,
    "start_char": 0,
    "end_char": 27,
    "page_start": null,
    "page_end": null,
    "section_path": "Deployment",
    "access_level": "internal"
  }],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

Another tenant's document is `404`, not `403`, so document existence stays
private.

### Phase 2 trade-offs

- **Characters, not tokens.** There is no tokenizer dependency yet; at roughly
  four characters per token, 1200 characters is a conservative bound for
  embedding models. This will be revisited against the actual model.
- **Overlap is best-effort within a hard size limit.** It may be shorter than
  configured to stop on a sentence or word boundary. Never exceeding the size
  limit matters more than hitting the overlap exactly.
- **No embedding-based semantic splitting yet.** It would require embeddings and
  adds another failure mode. The strategy interface exists so it can be
  compared against the recursive baseline later instead of being assumed better.
- **Chunks do not cross page or heading boundaries**, even when a sentence does.
  For citations, provenance wins; section context and overlap limit the loss.

---

## Design decisions (Phase 0)

- **App factory, not a module-level singleton for tests.** `create_app(settings,
  database=...)` makes dependencies explicit. Production still gets a module
  singleton in `main.py` for `uvicorn rag_engine.main:app`.
- **Dependencies behind protocols.** `DatabaseProtocol` is the only thing the
  API knows about; unit tests inject a fake, so the suite is hermetic. The same
  pattern will be applied to embeddings, the LLM, the re-ranker and Pinecone.
- **Pure ASGI middleware.** `BaseHTTPMiddleware` adds a task per request and has
  contextvar/streaming caveats. A ~50-line ASGI middleware gives one access log
  line per request, a validated request id, and a single place to catch
  unexpected exceptions.
- **Request ids are validated, not trusted.** A caller-supplied `X-Request-ID`
  is echoed only when it matches `[A-Za-z0-9._-]{1,64}`; otherwise a fresh id is
  generated. Untrusted values are never written to headers or logs.
- **Query strings are not logged.** Paths are. Query parameters can carry user
  input and PII, and access logs are not the right store for them.
- **Separate liveness and readiness.** A dead database must fail readiness (stop
  routing traffic) without failing liveness (restart the container), otherwise
  Kubernetes-style restarts loop without fixing anything.
- **Readiness never leaks driver errors.** The traceback is logged; the response
  says only `database unreachable`, because driver messages can embed DSNs.
- **JSON logs by default.** Logs are consumed by machines first. `console` is an
  explicit opt-in, and is what Compose uses for readability.
- **Idempotent logging setup.** Only handlers installed by this package are
  replaced, so test tooling and third-party instrumentation keep working.
- **Dev and runtime image targets.** The `dev` target installs editable with dev
  dependencies (Compose bind-mounts `src` for live reload); `runtime` installs a
  wheel, drops the source tree and runs unprivileged.
- **CI never needs third-party API keys.** External services are mocked at the
  interface boundary; PostgreSQL runs as a service container.

### Known trade-offs

- Compose publishes PostgreSQL on the host for developer convenience. That is a
  local-only default and must not be copied into production.
- The readiness probe opens a pooled connection per check. That is desirable
  (it proves the pool works) but it does mean health checks are not free under
  very high probe frequency.
- No authentication yet: there is no user identity to authorize, and adding a
  half-designed auth scheme now would be worse than adding it with the tenancy
  model in a later phase.

---

## Testing

```bash
make test-unit          # hermetic unit tests only
make lint               # ruff check + format check
make typecheck          # mypy (strict)

# Full suite + coverage gate; integration tests need a real PostgreSQL:
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://rag:rag_dev_password@localhost:55432/rag_test
make test
```

- Unit tests use `TestClient` with an injected fake database; they never open a
  socket.
- Integration tests run Alembic migrations once, truncate tables between tests,
  and exercise the app through its real lifespan. Without
  `RAG_TEST_DATABASE_URL` the whole directory is skipped.
- `docker compose exec api pytest` runs the full suite against the isolated
  `rag_test` database inside the stack.
- Coverage is enforced at **90%** (branch coverage) by `pyproject.toml`, with
  greenlet/thread concurrency enabled so lines after awaited SQLAlchemy calls
  are measured correctly.

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

1. **Quality** — `ruff check`, `ruff format --check`, `mypy` (strict).
2. **Tests** — Python 3.11 and 3.12, with a PostgreSQL 16 service container,
   coverage uploaded as an artifact.
3. **Container** — validates `docker-compose.yml` and builds the runtime image.

## Roadmap

| Phase | Scope |
| --- | --- |
| 0 | Foundation: skeleton, config, logging, DB, tests, CI, Compose ✅ |
| 1 | Ingestion: upload, parsing, chunking, async jobs, metadata, dedupe ✅ |
| 2 | Chunking: boundary-aware splits, source offsets, access metadata, inspection ✅ |
| 3 | Embeddings + Pinecone + BM25 hybrid retrieval with ACL filters |
| 4 | Re-ranking, grounded generation, citations |
| 5 | Guardrails and access-control enforcement |
| 6 | Evaluation harness, latency and irrelevant-context measurements |
