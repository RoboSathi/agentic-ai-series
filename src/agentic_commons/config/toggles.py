"""Feature toggles for agentic commons subsystems."""

from typing import Any

from agentic_commons.config.settings import get_settings


def is_enabled(subsystem: str) -> bool:
    """Check if a subsystem is enabled via config."""
    settings = get_settings()
    node: Any = settings
    for part in subsystem.split("."):
        node = getattr(node, part, None)
        if node is None:
            return False
    return bool(node)
