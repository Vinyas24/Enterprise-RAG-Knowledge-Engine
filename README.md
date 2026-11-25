# RAG Knowledge Engine

Enterprise-grade retrieval-augmented generation backend: ingest an
organization's documents, search them with hybrid retrieval, and answer
questions with grounded, cited responses — while enforcing tenant and access
control at retrieval time.

> **Status: Phase 5 — evaluation harness.**
> Documents are uploaded, parsed, chunked, embedded and indexed asynchronously;
> chunks are searchable by meaning, isolated per tenant and filtered by access
> clearance. `POST /query` returns an answer grounded in retrieved chunks with
> verifiable citations. Reindexing and deletion keep PostgreSQL and the vector
> store consistent, including on failure. A labeled retrieval benchmark now
> measures recall, MRR, precision, irrelevant context and latency against a
> saved dense-only baseline, so later retrieval changes are proven rather than
> assumed. Guardrails land in the next phase.

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
| `http://localhost:8000/documents/{id}/chunks` | Stored chunks with citation provenance |
| `http://localhost:8000/documents/{id}/reindex` | Re-run ingestion for a document |
| `http://localhost:8000/ingestion-jobs/{id}` | Ingestion job status and errors |
| `http://localhost:8000/query` | Retrieve, generate, and return citations |
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
├── embedding/      provider interface, batching, retries, OpenAI adapter
├── vectorstore/    store interface, memory + Pinecone, stale-vector cleanup
├── retrieval/      dense, BM25 keyword and hybrid retrieval with rank fusion
├── generation/     prompt assembly, LLM providers, citation mapping
├── evaluation/     dataset loader, metrics, isolated harness, CLI report
└── guardrails/     input checks, grounding, access control   (Phase 7+)
eval/
├── datasets/       committed corpus and labeled questions
├── baselines/      saved reports used for comparison
└── reports/        rendered reports committed with a phase
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
| `RAG_EMBEDDING_PROVIDER` | `deterministic` | `deterministic` (no key) or `openai` |
| `RAG_EMBEDDING_DIMENSIONS` | `256` | Must match the Pinecone index dimension |
| `RAG_EMBEDDING_BATCH_SIZE` | `64` | Texts per provider request |
| `RAG_EMBEDDING_MAX_RETRIES` | `4` | Retries with exponential backoff |
| `RAG_VECTOR_STORE` | `memory` | `memory` (tests/local) or `pinecone` |
| `RAG_PINECONE_INDEX_NAME` | `rag-engine` | Created on first use if absent |
| `RAG_PINECONE_CONTROL_HOST` | unset | Point at Pinecone Local in Compose |
| `RAG_RETRIEVAL_TOP_K` | `5` | Default candidates per query |
| `RAG_RETRIEVAL_MIN_SCORE` | `0.0` | Dense similarity floor (ignored by BM25) |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `dense` (Phase 4 behaviour) or `hybrid` |
| `RAG_HYBRID_DENSE_WEIGHT` | `1.0` | RRF weight for the dense ranking |
| `RAG_HYBRID_KEYWORD_WEIGHT` | `1.0` | RRF weight for the BM25 ranking |
| `RAG_HYBRID_RRF_K` | `60` | RRF damping constant |
| `RAG_RETRIEVAL_KEYWORD_CANDIDATES` | `200` | Full-text candidates BM25 re-ranks |
| `RAG_LLM_PROVIDER` | `deterministic` | `deterministic` (no key) or `openai` |

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

## Embeddings and vector storage (Phase 3)

Goal: a chunk becomes findable by meaning, not only by the words it contains,
while a document is never reported as searchable before its vectors exist.

**Provider interface.** `Embedder` is a Protocol (`dimensions`, `embed`,
`close`). `build_embedder(settings)` selects `deterministic` or `openai`.
The deterministic provider hashes tokens into a fixed-size vector, which makes
the full pipeline runnable offline and keeps CI free of API keys; it is a test
double, not a production embedding model. The OpenAI adapter batches inputs,
retries `429`/`5xx` with exponential backoff and jitter, and surfaces a typed
`EmbeddingError` for the worker to retry.

**Vector-store interface.** `VectorStore` is a Protocol (`ensure_ready`,
`upsert`, `delete`, `list_ids`, `query`, `close`). `MemoryVectorStore` is used
by the test suite; `PineconeVectorStore` is the production adapter. The
adapter creates the index on first use and validates the dimension and metric
instead of silently returning bad neighbours.

**Tenant isolation is structural.** Every tenant has its own Pinecone
namespace (`str(tenant_id)`). A missing metadata filter therefore cannot leak
another tenant's data. Metadata still carries `tenant_id` for defence in depth
and for future shared-namespace indexes.

**Ids are deterministic.** A vector id is `"<document_id>:<chunk_index>"` and
the PostgreSQL chunk id is a UUIDv5 of the same string (`core/ids.py`). A
retry upserts the same ids, so replacement is idempotent; a shrinking document
leaves no duplicate tail.

**Write ordering is fail-closed.** The worker embeds, upserts vectors, removes
stale vectors and only then commits chunks and marks the job `done`. If the
vector store fails, the transaction is not committed, the job is retried with
backoff, and the previous chunk/vector set remains searchable. The inverse
order — commit first, repair later — would mark a document indexed while it is
not.

**Reindex is explicit.** `POST /documents/{id}/reindex` queues a fresh pass for
the stored blob (useful after changing embedding model, chunk size or access
level). It takes a row lock so concurrent requests cannot enqueue duplicate
jobs, and returns `409` while a job is already queued or processing.

**Delete is fail-closed too.** `DELETE /documents/{id}` purges vectors first.
If the vector store is unavailable it returns `503` and keeps the document, so
a retry converges. It refuses with `409` while an ingestion job is active and
performs the final delete conditionally, so a job that becomes active between
the check and the commit cannot be orphaned.

### Phase 3 trade-offs

- **Two stores cannot be atomic.** A crash after the vector upsert but before
  the PostgreSQL commit leaves new vectors next to old chunks. Deterministic
  ids make the retry a replacement, and the stale-vector sweep (explicit old
  ids plus a prefix listing) removes anything the new chunk set does not keep.
- **Deletes enumerate ids instead of filtering metadata.** Not every Pinecone
  tier supports filtered deletes, and Pinecone Local rejects them. Prefix
  listing is portable; the trade-off is one extra list call per delete.
- **Vector listing can be eventually consistent.** Cleanup uses ids read from
  PostgreSQL first and treats the prefix sweep as a second chance, so a lagging
  list does not leave known vectors behind.
- **The embedding dimension is deployment-wide.** Changing the model or
  dimension requires a new index and a reindex; the adapter fails fast on a
  mismatch rather than writing unusable vectors.

---

## Retrieval and grounded generation (Phase 4)

Goal: answer a question from the caller's own documents, with citations a
reader can verify, and say so when the documents do not contain the answer.

**Retrieval.** `POST /query` embeds the question, searches the tenant's
namespace, hydrates the matching chunks from PostgreSQL, and discards any hit
whose document or chunk no longer exists. The search is bounded by
`top_k` and `min_score`; the namespace plus the hydrated `tenant_id` check
enforce isolation twice.

**Clearance filtering.** Access levels are ordered
`public < internal < confidential < restricted`. The request may carry a
clearance (default `internal`); retrieval only returns chunks at or below it. The filter is
applied in PostgreSQL against the snapshotted chunk `access_level`, so a
document whose level changed after ingestion cannot bypass it through stale
vector metadata.

**Generation.** `build_llm(settings)` selects a deterministic provider for
tests/CI or OpenAI for real answers. The prompt contains only the retrieved
chunks, numbered, with document titles and section/page provenance, and
instructs the model to answer solely from them and to admit when they are
insufficient.

**Citations are data, not prose.** Each answer is returned with the chunk id,
document id and title, chunk index, a snippet, page span and section path. The
same provenance is available from `GET /documents/{id}/chunks`, so a citation
can be checked against the stored chunk. `weak_retrieval` is `true` when no
chunk cleared the score floor or when the model reports that the context does
not answer the question, which lets a client distinguish "no answer in the
corpus" from a confident but unsupported answer.

```bash
curl -X POST http://localhost:8000/query \
  -H 'X-Tenant-ID: acme' -H 'Content-Type: application/json' \
  -d '{"question": "How do I roll back a deployment?", "top_k": 3}'
```

### Phase 4 trade-offs

- **Dense retrieval only.** BM25/hybrid fusion and a cross-encoder re-ranker
  are deliberately deferred. The Phase 5 benchmark now exists so each can be
  judged on measured quality and latency instead of intuition.
- **Single-shot generation.** No query rewriting, multi-hop or agent loop yet.
  One retrieval pass keeps latency and failure modes predictable.
- **Clearance is supplied by the caller.** Authentication is not implemented;
  the header-driven tenant and clearance model is the seam where real identity
  will plug in during Phase 6.
- **Citations are only as good as chunking.** The chunk span and section path
  make an answer auditable, but a chunk that mixes ideas will produce a vague
  citation. That is why chunking quality was treated as a first-class phase.

---

## Evaluation harness (Phase 5)

Goal: make retrieval quality measurable before changing retrieval. Every later
experiment (hybrid search, re-ranking, a different chunk size) is compared with
the same dataset, the same metric definitions and a saved baseline.

**Dataset.** `eval/datasets/enterprise_baseline.json` contains 10 enterprise
documents and 42 labeled questions. Each question lists its
`relevant_documents`, not chunk ids. Document-level labels are deliberately
coarser than chunk-level labels: they stay valid when chunking changes, so a
chunking experiment can be compared on identical ground truth. The manifest
supports inline content or a `path` relative to the manifest; the committed
corpus uses readable Markdown files in a sibling directory.

**One command.** The committed baseline was recorded with deterministic
embeddings, 256 dimensions, the in-memory store, default chunking
(1200 / 200 characters), `top_k=5` and `min_score=0.0`:

```bash
RAG_DATABASE_URL=postgresql+asyncpg://rag:rag_dev_password@localhost:55432/rag \
RAG_VECTOR_STORE=memory RAG_EMBEDDING_PROVIDER=deterministic \
RAG_EMBEDDING_DIMENSIONS=256 make eval EVAL_ARGS="--retriever dense"
```

(Phase 6 made hybrid the default, so `--retriever dense` is what reproduces
this baseline; see the Phase 6 section.)

`make eval` prints the Markdown report and automatically compares against
`eval/baselines/dense_baseline.json` when it exists. Pass extra flags through
`EVAL_ARGS`, for example:

```bash
make eval EVAL_ARGS="--json-output eval/reports/current.json \
  --save-baseline eval/baselines/dense_baseline.json"
```

The CLI also accepts `--dataset`, `--top-k`, `--min-score`, `--retriever`,
`--dense-weight`, `--keyword-weight`, `--rrf-k`, `--keyword-candidates`,
`--baseline` and `--markdown-output`. Evaluation needs only PostgreSQL and the
configured vector store; it does not need the API or worker to be running.

**Dense-only baseline** (`eval/baselines/dense_baseline.json`):

| Metric | Value |
| --- | ---: |
| Recall@5 | 0.9524 |
| Precision@5 | 0.2619 |
| MRR | 0.6865 |
| Hit rate | 0.9524 |
| Irrelevant-context rate | 0.7381 |
| Latency p50 | 8.18 ms |
| Latency p95 | 14.04 ms |

Latency is hardware-dependent; the quality metrics are reproducible with the
deterministic embedder. The baseline is a regression anchor for this project,
not a claim about production semantic quality.

**Metric definitions.** A retrieved chunk is relevant when its document key is
in the question's `relevant_documents`.

- **Recall@k**: unique expected documents found in the top k chunks divided by
  the number of expected documents, averaged over questions.
- **Precision@k**: relevant chunks in the top k divided by k, averaged over
  questions. Fewer than k results are not rounded up.
- **MRR**: reciprocal rank of the first relevant chunk (`0` when none),
  averaged over questions.
- **Hit rate**: share of questions with at least one relevant chunk.
- **Irrelevant-context rate**: irrelevant retrieved chunks divided by all
  retrieved chunks, micro-averaged across questions. This is the share of the
  generator's context that the labels do not consider relevant.
- **Latency**: wall time around `Retriever.retrieve` only — query embedding,
  vector search, keyword candidate selection and BM25 scoring, PostgreSQL
  hydration and clearance filtering, whichever of those the configured
  retriever runs — after one untimed warm-up. It excludes ingestion, dataset
  loading, metric calculation, report rendering and process startup. p50/p95 use
  the nearest-rank method.

**Isolation and repeatability.** Each run creates a throwaway tenant and blob
directory, indexes the corpus with the production chunker, embedder, ids and
vector metadata, and deletes the tenant and vectors afterwards even when a
question fails. Indexing is done directly rather than through the job queue, so
a worker running against the same database cannot claim evaluation jobs and
silently change the corpus. Chunking, embedding and vector construction still
come from the production modules. Retrieval breaks equal scores by document
title, chunk index and content, so a backend's arbitrary tie order cannot make
the quality metrics drift between runs.

**Staleness is explicit.** The report stores a dataset fingerprint and the
retrieval configuration. Comparing against a baseline with a different
fingerprint or configuration still works, but the report warns that the deltas
include corpus or configuration changes, not only retrieval changes.

### Phase 5 trade-offs

- **Document-level labels lose chunk-level nuance.** They cannot tell whether
  the best passage or a marginal passage from the right document was retrieved.
  The benefit is that labels survive chunking changes; chunk-level labels would
  need relabeling on every chunking experiment.
- **Precision@5 looks low by construction.** Most questions have one expected
  document and k=5, so at most one fifth of the returned chunks can be relevant
  when each document contributes one chunk. The metric is still useful for
  comparing runs; the irrelevant-context rate makes the prompt-level cost
  visible.
- **The deterministic embedder is not production quality.** It is a stable,
  offline feature hasher, which keeps the baseline reproducible and CI
  hermetic. Two customer-FAQ questions already miss under it; that is a useful
  floor, not evidence that the production model will behave the same way.
- **Ingestion is excluded from latency on purpose.** The metric answers "how
  long does a query wait for context?", not "how fast can the worker index a
  document?". Keeping both in one number would make neither actionable.

---

## Hybrid search (Phase 6)

Goal: recover queries dense retrieval misses without giving up semantic recall,
and measure the change against the Phase 5 baseline. Both retrieval paths use
the same tenant and clearance predicates, so hybrid search cannot widen what a
caller is allowed to read.

### Where the BM25 index lives: PostgreSQL

`chunks.content_tsv` is a generated `tsvector` column
(`to_tsvector('simple', content)`) with a GIN index, added by migration `0003`.
The database maintains it inside the same transaction that writes the chunk, so
a rolled-back ingestion leaves no keyword entries behind and no ingestion code
path can forget to update the index.

The alternatives were rejected for concrete reasons:

- **In-app index.** Would need a second write path, would drift from PostgreSQL
  on a failed or partially applied ingestion, and would not survive a process
  restart without a rebuild. Multi-worker deployments would need a shared
  index anyway, which is a database.
- **Pinecone sparse vectors.** Keeps one backend, but tenant and clearance
  filtering move into vector metadata filters, corpus-wide document frequencies
  are not available cheaply, and the keyword path would only be verifiable
  against Pinecone Local in this project.

The `simple` configuration is deliberate: it lowercases and splits on non-word
characters without stemming or stopword removal, so a lexeme is exactly a
tokenised word. Python tokenises with the same rule (`\w+`), which is what makes
per-chunk term frequency and per-term document frequency comparable.

### How a keyword query is scored

PostgreSQL selects candidates; BM25 is computed in Python over them.

1. Tokenise the query and deduplicate terms.
2. Read corpus size and mean `char_count`, plus each term's document frequency,
   from PostgreSQL — all inside the caller's tenant and clearance scope.
3. Drop terms that appear in at least half the visible corpus (and in at least
   two chunks) from *candidate selection*. With OR semantics they would match
   nearly every chunk and dilute the bounded candidate pool; BM25 already gives
   them an IDF near zero. If every term is common, the rarest one is kept so the
   query still produces candidates.
4. Select at most `RAG_RETRIEVAL_KEYWORD_CANDIDATES` chunks with
   `content_tsv @@ to_tsquery('simple', ...)`, ordered by `ts_rank`.
5. Score each candidate with BM25 (`k1=1.5`, `b=0.75`) using term frequencies
   from Python, document frequencies from PostgreSQL, and `char_count` as the
   document-length proxy.

Two details are deliberate. `ts_rank` is used only to decide which candidates
survive the cut — it is not BM25 (no corpus frequency, no length term) and is
roughly ten times cheaper than `ts_rank_cd`, which buys nothing when BM25
re-ranks anyway. And `char_count` stands in for token count: it reuses a column
ingestion already writes instead of adding one that would need a backfill.

### Fusion

Both paths run concurrently with `asyncio.gather`, each asked for
`max(top_k * 3, 20)` candidates. Results are merged with weighted reciprocal
rank fusion:

```
score(chunk) = Σ_paths weight_path / (k + rank_path)
```

Ranks, not scores, are fused: a cosine similarity and a BM25 score live on
unrelated scales, and normalising them would make the weights depend on the
result set. A zero weight removes that path from the fan-out entirely. Ties are
broken by document title, chunk index, content and chunk id, so metrics do not
drift between runs. `min_score` is forwarded to the dense path only; it is a
similarity floor and would discard every BM25 result if applied to it.

### Filtering

| Concern | Dense path | Keyword path |
| --- | --- | --- |
| Tenant | vector namespace + `chunks.tenant_id` | `chunks.tenant_id` |
| Clearance | vector metadata + `chunks.access_level` | `chunks.access_level` |
| Score floor | cosine `min_score` | not applicable |

The keyword path filters once, in the query that selects candidates. Its corpus
statistics are computed in the same visible scope: using the whole table would
let restricted chunks shift the score of chunks a caller can read, which leaks
their existence. The cost is that BM25 scores are not comparable across
clearance levels — intentional, because comparing them would leak the same
information.

### Result: hybrid versus the dense-only baseline

Committed report: `eval/reports/hybrid_vs_dense.md` (JSON alongside it). Same
dataset fingerprint (`8a0bf1e6…`), same chunking, same deterministic embedder,
`top_k=5`, `min_score=0.0`, weights `1.0 / 1.0`, `k=60`.

| Metric | Dense baseline | Hybrid | Δ |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.9524 | **1.0000** | +0.0476 |
| Precision@5 | 0.2619 | **0.3429** | +0.0810 |
| MRR | 0.6865 | **0.9095** | +0.2230 |
| Hit rate | 0.9524 | **1.0000** | +0.0476 |
| Irrelevant-context rate | 0.7381 | **0.6571** | −0.0810 |
| Latency p50 | 8.18 ms | 24.56 ms | +16.38 ms |
| Latency p95 | 14.04 ms | 35.02 ms | +20.98 ms |

Hybrid retrieval fixes both dense misses (`faq-01` and `faq-04`, both on the
customer FAQ) and moves the first relevant chunk from median rank ~1.5 to ~1.1.
Quality metrics are reproducible with the deterministic embedder; latency is
not. On the local Docker host, p50 across repeated runs ranged from roughly 20
to 30 ms (and dense from 8 to 9 ms), so treat the latency delta as "roughly two
to three times", not a precise multiplier. The cause is explained under
trade-offs.

The report warns that the baseline configuration differs. That is the point:
the delta measures the retriever change, and the report says so rather than
presenting it as a like-for-like comparison.

**Weights are tunable, but not tuned on the test set.** A sweep on the same 42
questions:

| Dense : keyword | Recall@5 | Precision@5 | MRR | Irrelevant-context |
| --- | ---: | ---: | ---: | ---: |
| 1 : 1 (default) | 1.0000 | 0.3429 | 0.9095 | 0.6571 |
| 2 : 1 | 0.9762 | 0.3143 | 0.8651 | 0.6857 |
| 1 : 2 | 1.0000 | 0.3476 | 0.9393 | 0.6524 |
| 3 : 1 | 0.9762 | 0.3048 | 0.8333 | 0.6952 |
| 1 : 3 | 0.9762 | 0.3524 | 0.9333 | 0.6476 |

Keyword-heavier weighting scores slightly better here, but with 42 questions
that gap is not distinguishable from noise. The default stays neutral at 1:1
and the knob is exposed for a corpus that justifies tuning.

### Reproducing

```bash
# Phase 6 comparison: hybrid run against the saved dense baseline.
RAG_DATABASE_URL=postgresql+asyncpg://rag:rag_dev_password@localhost:55432/rag \
RAG_VECTOR_STORE=memory RAG_EMBEDDING_PROVIDER=deterministic \
RAG_EMBEDDING_DIMENSIONS=256 make eval

# Dense-only, to confirm the Phase 5 baseline still reproduces.
make eval EVAL_ARGS="--retriever dense"

# Compare against the hybrid anchor instead (regression testing Phase 6 itself).
make eval EVAL_ARGS="--baseline eval/baselines/hybrid_baseline.json"

# Weight and fusion experiments.
make eval EVAL_ARGS="--dense-weight 1.0 --keyword-weight 2.0 --rrf-k 60"
```

`make` defaults to the ambient `python3`, which is often not the interpreter
that has `rag_engine` installed. With the project virtualenv present but not
active, pass it explicitly: `make eval PYTHON=.venv/bin/python`. The same
applies to every `make eval` command above.

### Phase 6 trade-offs

- **No stemming.** `simple` keeps Python and PostgreSQL tokenisation identical,
  but "rollback" and "roll back" are different lexemes. A stemmed
  configuration would match more, and would then need the same stemmer in
  Python for term frequency — or PostgreSQL would have to compute term
  frequency, and `tsvector_to_array` discards positions and repeats. Matching
  behaviour beat extra recall for this phase.
- **BM25 is exact only inside the candidate pool.** With more than
  `RAG_RETRIEVAL_KEYWORD_CANDIDATES` matches, the cut is decided by `ts_rank`,
  a cheaper heuristic. The default of 200 is far above `top_k=5` on the
  evaluation corpus, so the evaluation never exercises the cut; a large tenant
  would.
- **`char_count` is a length proxy, not a token count.** It is already stored
  and maintained, which avoided a migration and a backfill. A chunk with many
  short words is treated as longer than it is.
- **Hybrid latency is roughly 2–3× dense.** Two backends, three extra
  PostgreSQL round trips for the keyword path, and a 3× candidate overfetch per
  path. On the local Docker host the two paths also contend for database CPU,
  so the parallel fan-out hides less than it would with headroom. The
  alternative — a keyword-only path for identifier queries — would need a
  query classifier this phase does not have.
- **RRF throws away score magnitude.** A chunk that is a near-perfect dense
  match and one that barely made the top 20 contribute the same at a given
  rank. That is the price of not calibrating two unrelated scales; a
  score-fusion experiment would need per-backend normalisation and its own
  baseline.
- **Scores in `per_question` are fused RRF scores.** They are useful for
  ordering and for spotting rank changes, not for thresholding. The report
  states this so a future reader does not compare them with cosine values.
- **Corpus statistics are clearance-scoped.** Two users with different
  clearances get different BM25 scores for the same chunk. That is the
  deliberate choice over leaking the existence of restricted documents.

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
make eval               # retrieval metrics vs. the saved dense baseline

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
| 3 | Embeddings + Pinecone, tenant namespaces, consistent reindex/delete ✅ |
| 4 | Dense retrieval with clearance filters, grounded generation, citations ✅ |
| 5 | Evaluation harness: labeled corpus, recall/MRR/precision, latency, saved dense baseline ✅ |
| 6 | Hybrid retrieval: PostgreSQL BM25 index, parallel fan-out, weighted RRF, measured vs Phase 5 ✅ |
| 7 | Guardrails and access-control enforcement |
