# ADR 1: Boundary-Aware Recursive Document Chunking

- **Status:** Accepted
- **Deciders:** Antigravity Engineering Team
- **Date:** 2026-11-26

## Context & Problem Statement

Retrieval-augmented generation depends critically on the quality and integrity of chunked passages. Fixed-width character or token windowing arbitrarily bisects sentences, code blocks, bulleted lists, and paragraphs. Furthermore, when chunks cross structural boundaries (such as page breaks in PDFs or section headings in Markdown/DOCX), downstream citation tracking breaks down because a chunk no longer points to a single verifiable source location.

How should incoming documents of varied formats (PDF, DOCX, Markdown, plain text) be split into high-quality, traceable chunks?

## Decision

We chose to implement boundary-aware recursive chunking (`RecursiveChunker`) behind a strategy protocol (`ChunkingStrategy`):

1. **Strict Structural Isolation:** Chunks never cross document structural boundaries (page breaks in PDFs, heading hierarchy in Markdown/DOCX). Each parsed block is chunked independently.
2. **Recursive Delimiter Hierarchy:** When a block exceeds `RAG_CHUNK_SIZE_CHARS` (default 1200 chars), the splitter recursively tests delimiters in descending semantic order:
   - Paragraph break (`\n\n`)
   - Line break (`\n`)
   - Sentence termination punctuation (`. `, `! `, `? `)
   - Whitespace (` `)
   - Hard character cutoff as a last resort
3. **Overlap Preservation:** Adjacent chunks within the same structural unit retain configurable character overlap (`RAG_CHUNK_OVERLAP_CHARS`, default 200 chars) to prevent context cliffing at chunk boundaries.
4. **Source Provenance Offsets:** Every generated chunk records exact source offsets: character span (`start_char`, `end_char`), page span (`page_start`, `page_end`), heading path (`section_path`), and document access level.

## Consequences

### Positive
- **Verifiable Citations:** Downstream generation can cite exact pages and sections because every chunk originates from a bounded structural span.
- **Semantic Cohesion:** Sentences and paragraphs are kept intact wherever possible, maximizing dense embedding semantic representation and BM25 token overlap.
- **Pluggability:** The `ChunkingStrategy` interface allows alternative strategies (such as semantic chunking or token-based chunking) to be benchmarked without altering ingestion worker persistence logic.

### Negative & Trade-offs
- **Variable Chunk Sizes:** Chunks are not uniform in length; small paragraphs produce smaller chunks, occasionally resulting in minor vector density variance.
- **Scanned Documents:** Documents without clean textual layers (e.g. scanned image PDFs) produce unparseable blocks and are explicitly rejected with `415 Unsupported Media Type` rather than producing empty chunks.

## Alternatives Considered
- **Fixed-Width Token Splitters (e.g. LangChain CharacterTextSplitter):** Rejected due to high risk of cutting sentences in half and inability to respect document page/section boundaries.
- **Semantic LLM-based Chunking:** Rejected due to cost and latency overhead during high-volume document ingestion.
