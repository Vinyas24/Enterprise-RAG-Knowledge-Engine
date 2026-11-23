"""Enterprise RAG knowledge engine.

The package is split into vertical slices that mirror the pipeline:

- ``ingestion``  document intake, parsing, chunking, indexing
- ``retrieval``  dense + keyword retrieval and candidate fusion
- ``ranking``    re-ranking of merged candidates
- ``generation`` prompt construction and cited answer generation
- ``guardrails`` input/output safety and access-control enforcement
- ``core``       configuration, logging, database plumbing
- ``api``        HTTP transport layer (routers, middleware, error envelopes)

Import direction is one-way: ``api`` -> domain slices -> ``core``. Domain
slices never import from ``api``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
