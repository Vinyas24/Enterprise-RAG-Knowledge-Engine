# Architecture Decision Records (ADRs)

This directory documents the key architectural decisions made during the design
and evolution of the RAG Knowledge Engine.

| ADR | Title | Status | Scope |
| --- | --- | --- | --- |
| [0001](0001-chunking-strategy.md) | Boundary-Aware Recursive Document Chunking | Accepted | Ingestion, Parsers, Chunking |
| [0002](0002-hybrid-retrieval.md) | Hybrid Retrieval with Parallel Fan-Out and RRF | Accepted | Retrieval, PostgreSQL BM25, Fusion |
| [0003](0003-reranking.md) | Two-Stage Retrieval with Candidate Over-Fetching and Fail-Open Re-Ranking | Accepted | Ranking, Precision, Cohere / Cross-Encoder |
| [0004](0004-multitenancy-and-access-control.md) | Multi-Tenancy Isolation and Lattice Access Control | Accepted | Security, Clearance, Tenant Namespacing |
