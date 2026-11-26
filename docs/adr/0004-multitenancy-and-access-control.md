# ADR 4: Multi-Tenancy Isolation and Lattice Access Control

- **Status:** Accepted
- **Deciders:** Antigravity Engineering Team
- **Date:** 2026-11-26

## Context & Problem Statement

In enterprise multi-tenant deployments, distinct organizations share the RAG infrastructure. Data cross-contamination between tenants (e.g. Acme Corp retrieving Globex Corp's financial filings) represents a catastrophic security failure. Furthermore, within a tenant, documents possess differing confidentiality levels (e.g., an intern retrieving executive compensation documents).

How should tenant isolation and user access permissions be strictly enforced across database, vector store, and caching layers?

## Decision

We chose to implement a defense-in-depth multi-tenant isolation and lattice-based clearance model:

1. **Relational Isolation:**
   - Every `documents`, `chunks`, and `ingestion_jobs` row stores a foreign key to `tenants.id`.
   - All queries and mutations strictly mandate `WHERE tenant_id == :tenant_id`. Foreign key cascade rules ensure clean deletion.
2. **Vector Store Namespacing:**
   - Vector embeddings are indexed under dedicated tenant namespaces (`namespace=str(tenant.id)`). Queries to Pinecone or in-memory vector stores mandate the namespace filter, isolating dense vector indices at the storage engine level.
3. **Lattice Clearance Hierarchy:**
   - Access levels form a strict mathematical poset:
     $$\text{PUBLIC} \le \text{INTERNAL} \le \text{CONFIDENTIAL} \le \text{RESTRICTED}$$
   - When a caller queries with clearance $C$, candidates are filtered by allowed levels:
     $$\text{allowed\_levels} = \{L \mid \text{rank}(L) \le \text{rank}(C)\}$$
   - Enforced consistently across dense vector search, BM25 keyword search, and document listing.
4. **Tenant-Partitioned Caching:**
   - Embedding and retrieval cache keys incorporate `tenant_id`.
   - When documents are deleted or re-indexed, `invalidate_tenant(tenant_id)` purges all cached results for that tenant instantly.
5. **Red-Team Verification:**
   - Verified against a 15-case red-team adversarial suite (`tests/integration/test_red_team.py` and `tests/integration/test_access_control.py`), proving zero cross-tenant leakage even with identical queries.

## Consequences

### Positive
- **Guaranteed Isolation:** Cross-tenant leakage is provably prevented across PostgreSQL, Pinecone, and in-memory caches.
- **Hierarchical Access Control:** Chunks exceeding the caller's clearance are filtered before ranking and context assembly, preventing sensitive data leakage into LLM prompts.
- **Auditability:** Every retrieval and clearance decision is recorded with tenant attribution in `audit_logs`.

### Negative & Trade-offs
- **Header Requirement:** All API data requests must supply `X-Tenant-ID` (or authenticated tenant context).
- **Index Fragmentation:** Partitioning vector spaces into small namespaces can reduce cluster-level vector batching efficiency compared to a flat unpartitioned index.

## Alternatives Considered
- **Database-per-Tenant:** Rejected due to excessive operational complexity and connection pooling overhead for thousands of tenants.
- **Application-Level Post-Filtering (Filtering out chunks after retrieval):** Rejected as dangerous and inefficient: if top-k candidates are all filtered out post-retrieval, the prompt receives empty context, and embedding distance calculations for unauthorized chunks waste CPU.
