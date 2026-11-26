# ADR 2: Hybrid Retrieval with Parallel Fan-Out and Reciprocal Rank Fusion

- **Status:** Accepted
- **Deciders:** Antigravity Engineering Team
- **Date:** 2026-11-26

## Context & Problem Statement

Dense vector retrieval (using embedding distance like cosine similarity) excels at capturing conceptual and semantic similarity, but struggles with exact token matches: domain acronyms, product IDs, version numbers, SKU codes, and rare technical keywords. Conversely, keyword retrieval (BM25) excels at lexical and token precision but fails on synonyms and paraphrasing.

How should the engine combine semantic and keyword search, where should the keyword index live, and how should candidate rankings be merged?

## Decision

We chose to implement a parallel hybrid retrieval architecture (`HybridRetriever`):

1. **Index Storage:**
   - **Dense Store:** Vector database (Pinecone in production / `MemoryVectorStore` in test).
   - **Keyword Index:** PostgreSQL full-text search utilizing `tsvector` with the `english` text search configuration, backed by a GIN index on `chunks.tsv` and evaluated using `ts_rank_cd` and `websearch_to_tsquery`.
2. **Parallel Fan-Out:**
   - Queries dispatch dense vector search and PostgreSQL BM25 search concurrently via `asyncio.gather`.
   - Both paths enforce identical tenant (`WHERE tenant_id == :tenant_id`) and clearance filters.
3. **Reciprocal Rank Fusion (RRF):**
   - Candidate rankings are merged using weighted Reciprocal Rank Fusion:
     $$RRF(d) = \sum_{m \in M} w_m \cdot \frac{1}{k + r_m(d)}$$
   - Default constant $k=60$, with tunable path weights ($w_{\text{dense}}=1.0, w_{\text{keyword}}=1.0$).
4. **Outage Resilience & Fallback:**
   - If either the dense store or PostgreSQL keyword search encounters a timeout or transient failure, `HybridRetriever` logs a warning and falls back to the surviving path rather than failing the user query.

## Consequences

### Positive
- **Proven Quality Gains:** Empirical evaluation showed MRR improved from 0.8810 (dense-only) to 0.9095 (hybrid), and Precision@5 improved from 0.3143 to 0.3429.
- **Operational Simplicity:** Utilizing PostgreSQL for BM25 avoided deploying and maintaining an external Elasticsearch or OpenSearch cluster.
- **Score Scale Invariance:** RRF operates purely on candidate ranks, eliminating the need to calibrate or normalize heterogeneous dense cosine similarities (bounded $[0, 1]$) with unbounded BM25 scores.

### Negative & Trade-offs
- **Candidate Over-fetching:** Each search path must over-fetch candidates (multiplier $3\times$, minimum 20) to give rank fusion sufficient overlap to promote multi-path matches.
- **PostgreSQL CPU Overhead:** High-frequency full-text queries place load on the primary relational database, which is mitigated by query caching and GIN indexing.

## Alternatives Considered
- **Dedicated Search Cluster (Elasticsearch/OpenSearch):** Rejected to avoid adding another heavy operational dependency when PostgreSQL native full-text search meets latency and quality goals.
- **Pinecone Sparse Vectors (SPLADE):** Rejected to prevent vendor lock-in and retain lexical index control within the relational schema.
- **Linear Score Combination:** Rejected due to score scale instability across query lengths and corpus changes.
