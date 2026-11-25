# Retrieval evaluation — enterprise-baseline

Generated: 2026-11-25T17:40:08.783507+00:00
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

## Warnings

- baseline configuration differs: hybrid_dense_weight, hybrid_keyword_weight, hybrid_rrf_k, keyword_candidates
- baseline used retriever 'dense' and this run used 'hybrid'; the delta is the effect of that change, not a like-for-like comparison

## Metrics

| Metric | Current | Baseline | Δ | Direction |
| --- | ---: | ---: | ---: | --- |
| Recall@k | 1.0000 | 0.9524 | +0.0476 | higher is better |
| Precision@k | 0.3429 | 0.2619 | +0.0810 | higher is better |
| MRR | 0.9095 | 0.6865 | +0.2230 | higher is better |
| Hit rate | 1.0000 | 0.9524 | +0.0476 | higher is better |
| Irrelevant-context rate | 0.6571 | 0.7381 | -0.0810 | lower is better |
| Latency p50 (ms) | 24.56 | 8.18 | +16.38 | lower is better |
| Latency p95 (ms) | 35.02 | 14.04 | +20.98 | lower is better |
| Latency mean (ms) | 26.46 | 8.99 | +17.47 | lower is better |

## Misses and partial recalls

None — every question retrieved at least one chunk from every expected document.

## Metric definitions

A retrieved chunk is **relevant** when its document key is listed in the question's `relevant_documents`. Labels are document-level so they survive chunking changes.

- **Recall@5**: unique expected documents found in the top 5 chunks, divided by the number of expected documents; averaged over questions.
- **Precision@5**: relevant chunks in the top 5, divided by 5; averaged over questions. Fewer than k results are not rounded up.
- **MRR**: reciprocal rank of the first relevant chunk (0 when none); averaged.
- **Hit rate**: share of questions with at least one relevant chunk.
- **Irrelevant-context rate**: irrelevant retrieved chunks divided by all retrieved chunks, micro-averaged across questions. This is the share of context handed to the generator that the labels do not consider relevant.
- **Latency**: wall time around `Retriever.retrieve` only — query embedding, vector search, keyword candidate selection and BM25 scoring, PostgreSQL hydration and clearance filtering, whichever of those the configured retriever runs. Measured after one warm-up retrieval. It excludes ingestion, dataset loading, metric calculation, report rendering and process startup. p50/p95 use the nearest-rank method.

Retrieval path: weighted reciprocal rank fusion of dense and BM25 keyword results; no re-ranker is applied. Scores in `per_question` are fused RRF scores, so they are not comparable with cosine similarities.

