"""Domain vocabulary shared by persistence, API and (later) retrieval.

These enums are the single source of truth for stored string values. They are
persisted as VARCHAR + CHECK constraints rather than native PostgreSQL enums:
adding a value then does not require an ``ALTER TYPE`` migration, while the
database still rejects invalid values written outside the application.
"""

from __future__ import annotations

from enum import StrEnum


class AccessLevel(StrEnum):
    """Document sensitivity, ordered from least to most restrictive."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# Ordered by increasing sensitivity. Retrieval will expand a user's clearance
# into the set of levels they may see; keeping the order in one place prevents
# a "confidential" document from leaking to an "internal" user by accident.
ACCESS_LEVEL_ORDER: tuple[AccessLevel, ...] = (
    AccessLevel.PUBLIC,
    AccessLevel.INTERNAL,
    AccessLevel.CONFIDENTIAL,
    AccessLevel.RESTRICTED,
)


def visible_access_levels(clearance: AccessLevel) -> tuple[AccessLevel, ...]:
    """Access levels a caller with ``clearance`` is allowed to read."""
    return ACCESS_LEVEL_ORDER[: ACCESS_LEVEL_ORDER.index(clearance) + 1]


class DocumentStatus(StrEnum):
    """Lifecycle of a document from the uploader's point of view."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class JobStatus(StrEnum):
    """Lifecycle of a single ingestion attempt."""

    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class UserRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
