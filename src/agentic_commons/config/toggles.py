"""Feature toggles for agentic commons subsystems."""

from typing import Any

from agentic_commons.config.settings import get_settings


def is_enabled(subsystem: str) -> bool:
    """Check if a subsystem is enabled via config.

    Accepts either a bare subsystem path ("security.pii") or one ending in
    ".enabled" ("security.pii.enabled") — both resolve to the same toggle.
    A bare path resolves to a settings object, not a bool, so it falls
    back to that object's own `enabled` field rather than `bool(object)`
    (which is always True for a pydantic model, regardless of its fields).
    """
    settings = get_settings()
    node: Any = settings
    for part in subsystem.split("."):
        node = getattr(node, part, None)
        if node is None:
            return False
    if isinstance(node, bool):
        return node
    return bool(getattr(node, "enabled", False))
