"""chunk full-text index for keyword retrieval

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-25 10:00:00.000000+00:00

Phase 6 adds a keyword retrieval path next to dense vector search. The keyword
index lives in PostgreSQL rather than in the application or the vector store:

* PostgreSQL already holds every chunk, so a keyword index cannot drift from
  the source of truth and needs no second write path at ingestion time.
* A generated column keeps the index definition declarative: it cannot be
  forgotten by an ingestion code path, and reindexing is one ``UPDATE``.
* The ``simple`` text-search configuration is deliberate. It lowercases and
  splits on non-word characters but does not stem or drop stopwords, so a
  lexeme is exactly a tokenised word. BM25 term frequencies are computed from
  the same tokenisation in Python, which keeps the two halves of the score
  consistent.

``to_tsvector`` is only immutable when the configuration is passed explicitly,
which is why the column pins ``'simple'::regconfig`` instead of relying on
``default_text_search_config``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple'::regconfig, content)", persisted=True),
            nullable=False,
        ),
    )
    # GIN is the right operator class for ``@@`` containment queries; the
    # candidate query filters and ranks, so an index-only scan is not required.
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
