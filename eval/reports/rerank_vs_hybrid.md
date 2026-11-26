# Retrieval evaluation — enterprise-baseline

Generated: 2026-11-26T06:27:10.827616+00:00
Dataset fingerprint: `8a0bf1e6a64673537b3942127c465b1a1fa6c9ac0989a42ebc401ca06302351a`

## Corpus and configuration

- Documents: 10
- Indexed chunks: 50
- Questions: 42
- Retriever: hybrid (deterministic, 256 dimensions)
- Chunking: 1200 chars / 200 overlap
- Vector store: memory
- top_k: 5, min_score: 0.0
- Fusion: weighted reciprocal rank fusion (dense 1.0, keyword 1.0, k=60); keyword candidates: 200
- Re-ranker: lexical; 20 candidates → top 5, timeout 5s

## Warnings

- baseline used reranker 'off' and this run used 'lexical'; the delta is the effect of re-ranking, not a like-for-like comparison

## Metrics

| Metric | Current | Baseline | Δ | Direction |
| --- | ---: | ---: | ---: | --- |
| Recall@k | 1.0000 | 1.0000 | +0.0000 | higher is better |
| Precision@k | 0.3952 | 0.3429 | +0.0524 | higher is better |
| MRR | 0.9762 | 0.9095 | +0.0667 | higher is better |
| Hit rate | 1.0000 | 1.0000 | +0.0000 | higher is better |
| Irrelevant-context rate | 0.6048 | 0.6571 | -0.0524 | lower is better |
| Latency p50 (ms) | 9.38 | 25.44 | -16.06 | lower is better |
| Latency p95 (ms) | 18.58 | 50.15 | -31.57 | lower is better |
| Latency mean (ms) | 11.05 | 28.57 | -17.52 | lower is better |
| Re-rank latency p50 (ms) | 0.3047 | 0.0000 | +0.3047 | lower is better |
| Re-rank latency p95 (ms) | 0.6872 | 0.0000 | +0.6872 | lower is better |
| Re-rank latency mean (ms) | 0.3714 | 0.0000 | +0.3714 | lower is better |

Re-rank fallbacks: 0 of 42 questions returned the un-reranked order.

## Misses and partial recalls

None — every question retrieved at least one chunk from every expected document.

## Metric definitions

A retrieved chunk is **relevant** when its document key is listed in the question's `relevant_documents`. Labels are document-level so they survive chunking changes.

- **Recall@5**: unique expected documents found in the top 5 chunks, divided by the number of expected documents; averaged over questions.
- **Precision@5**: relevant chunks in the top 5, divided by 5; averaged over questions. Fewer than k results are not rounded up.
- **MRR**: reciprocal rank of the first relevant chunk (0 when none); averaged.
- **Hit rate**: share of questions with at least one relevant chunk.
- **Irrelevant-context rate**: irrelevant retrieved chunks divided by all retrieved chunks, micro-averaged across questions. This is the share of context handed to the generator that the labels do not consider relevant.
- **Latency**: wall time around `Retriever.retrieve` only — query embedding, vector search, keyword candidate selection and BM25 scoring, PostgreSQL hydration, clearance filtering and re-ranking when enabled, whichever of those the configured stack runs. Measured after one warm-up retrieval. It excludes ingestion, dataset loading, metric calculation, report rendering and process startup. p50/p95 use the nearest-rank method.
- **Re-rank latency**: time spent inside the re-ranker, a subset of the total latency above; zero when re-ranking is disabled. A timeout bills the full deadline before the fallback returns the un-reranked order.

Retrieval path: weighted reciprocal rank fusion of dense and BM25 keyword results, then lexical re-ranking of the merged candidates down to top_k. Scores in `per_question` are re-ranker relevance scores, so they are not comparable across providers or with cosine similarities.

