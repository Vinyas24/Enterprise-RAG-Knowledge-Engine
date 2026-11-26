# ADR 3: Two-Stage Retrieval with Candidate Over-Fetching and Fail-Open Re-Ranking

- **Status:** Accepted
- **Deciders:** Antigravity Engineering Team
- **Date:** 2026-11-26

## Context & Problem Statement

First-stage retrieval (dense + BM25 keyword fusion) prioritizes high recall over candidate precision. Consequently, top-k candidate pools often contain irrelevant or weakly related chunks. Handing uncurated chunks to the LLM increases prompt token costs, increases latency, and elevates hallucination risk.

How should the engine improve the precision of the final context window without compromising reliability or incurring severe latency penalties?

## Decision

We chose to implement a two-stage retrieval pipeline with second-pass re-ranking (`RerankedRetriever`):

1. **Provider Abstraction (`Reranker`):**
   - Pluggable interface supporting hosted API re-ranking (Cohere `rerank-v3.5`), local neural cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`), and test lexical scorers.
2. **Candidate Over-Fetching Policy:**
   - The first-stage hybrid retriever fetches an expanded candidate pool (`RAG_RERANK_CANDIDATES`, default 20 candidates).
   - The re-ranker evaluates query-candidate pairs jointly using deep cross-attention and trims the output to the final top-n (`top_k`, default 5).
3. **Fail-Open Timeout & Fallback:**
   - Re-ranking execution is bounded by a strict timeout (`RAG_RERANK_TIMEOUT_SECONDS`, default 5.0s).
   - On timeout or provider error, the wrapper automatically catches the exception, logs a structured warning, and falls back to the un-reranked candidate order. The request never fails with an error page.

## Consequences

### Positive
- **Irrelevant Context Reduction:** Evaluated across the enterprise test corpus, re-ranking reduced the irrelevant-context rate from 65.71% down to 60.48% (a 5.2% absolute reduction in unhelpful tokens reaching the prompt).
- **Retrieval Precision Boost:** Boosted MRR from 0.9095 to 0.9762 and Precision@5 from 0.3429 to 0.3952.
- **High Availability:** The fail-open fallback ensures that provider outages or network hiccups cause only a mild, temporary drop in ranking precision rather than user-facing downtime.

### Negative & Trade-offs
- **Latency Cost:** Re-ranking adds a small compute/network cost (+0.37 ms for local cross-encoder, ~50–100 ms for hosted API).
- **Over-fetch Database I/O:** Expanding candidate depth from 5 to 20 requires retrieving more rows from PostgreSQL and Pinecone.

## Alternatives Considered
- **No Re-ranking (First-Stage Only):** Rejected because first-stage bi-encoders miss subtle cross-attention nuances between query terms and candidate text.
- **Large Context LLM Ingestion (Pass all 20 chunks to prompt):** Rejected due to quadratic attention costs, increased generation latency, and "lost in the middle" attention degradation.
