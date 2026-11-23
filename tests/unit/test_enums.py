"""Domain enums: stored values and the access-level ordering rule."""

from __future__ import annotations

import pytest

from rag_engine.core.enums import (
    AccessLevel,
    DocumentStatus,
    JobStatus,
    UserRole,
    visible_access_levels,
)


def test_status_values_match_the_public_contract() -> None:
    assert [status.value for status in JobStatus] == ["queued", "processing", "done", "failed"]
    assert [status.value for status in DocumentStatus] == [
        "pending",
        "processing",
        "indexed",
        "failed",
    ]
    assert [role.value for role in UserRole] == ["owner", "admin", "member"]


@pytest.mark.parametrize(
    ("clearance", "visible"),
    [
        (AccessLevel.PUBLIC, ["public"]),
        (AccessLevel.INTERNAL, ["public", "internal"]),
        (AccessLevel.CONFIDENTIAL, ["public", "internal", "confidential"]),
        (
            AccessLevel.RESTRICTED,
            ["public", "internal", "confidential", "restricted"],
        ),
    ],
)
def test_clearance_expands_to_every_less_sensitive_level(
    clearance: AccessLevel, visible: list[str]
) -> None:
    assert [level.value for level in visible_access_levels(clearance)] == visible
