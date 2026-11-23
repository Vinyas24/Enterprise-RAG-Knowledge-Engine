"""chunk provenance and access metadata

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-23 15:20:00.000000+00:00

Chunks gain three pieces of Phase 2 metadata:

* ``access_level`` is copied from the document at ingestion time, so retrieval
  can enforce clearance without joining ``documents``.
* ``start_char``/``end_char`` point back into the parsed text, which makes a
  chunk's citation verifiable byte-for-byte.

Existing rows inherit their document's access level. Offsets stay ``NULL``:
Phase 1 did not record them, and inventing a range would be a lie.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Default first, so a concurrent insert during the migration cannot create
    # a chunk without an access level. The default is removed afterwards to
    # keep the application as the only source of the value.
    op.add_column(
        "chunks",
        sa.Column(
            "access_level",
            sa.String(length=16),
            server_default=sa.text("'internal'"),
            nullable=False,
        ),
    )
    op.add_column("chunks", sa.Column("start_char", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("end_char", sa.Integer(), nullable=True))

    op.execute(
        """
        UPDATE chunks
        SET access_level = (
            SELECT documents.access_level
            FROM documents
            WHERE documents.id = chunks.document_id
        )
        """
    )
    op.alter_column(
        "chunks", "access_level", existing_type=sa.String(length=16), server_default=None
    )

    op.create_check_constraint(
        "access_level",
        "chunks",
        "access_level IN ('public', 'internal', 'confidential', 'restricted')",
    )
    op.create_index(
        "ix_chunks_tenant_access_level",
        "chunks",
        ["tenant_id", "access_level"],
        unique=False,
    )
    op.create_check_constraint(
        "char_range",
        "chunks",
        "start_char IS NULL OR end_char IS NULL "
        "OR (start_char >= 0 AND end_char > start_char)",
    )


def downgrade() -> None:
    # Pass the bare names: the ``ck`` naming convention expands them to
    # ``ck_chunks_*`` exactly as it did when they were created.
    op.drop_constraint("char_range", "chunks", type_="check")
    op.drop_index("ix_chunks_tenant_access_level", table_name="chunks")
    op.drop_constraint("access_level", "chunks", type_="check")
    op.drop_column("chunks", "end_char")
    op.drop_column("chunks", "start_char")
    op.drop_column("chunks", "access_level")
