"""Loading the host a deployment names (`module.path:attribute`).

Which agent this process operates is a deployment decision, made in
configuration (`agent.host`), so the shell resolves it by name rather
than by import.
"""

import importlib
from typing import Any

from agent_platform.interface.host.protocols import AgentHost

# `module.path:attribute` — the same shape as a console-script entry
# point, because that is what this is.
SPEC_SEPARATOR = ":"


def load_spec(spec: str) -> Any:
    """Load the object named by a `module.path:attribute` string.

    Called when the attribute is callable (a class or factory), so a spec
    can name either an instance or the thing that builds one. Raises
    rather than falling back: a misspelled path must not quietly load
    nothing.
    """
    module_path, separator, attribute = spec.partition(SPEC_SEPARATOR)
    if not separator or not attribute:
        raise ValueError(f"'{spec}' is not a 'module.path{SPEC_SEPARATOR}attribute' spec.")
    module = importlib.import_module(module_path)
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"'{module_path}' has no attribute '{attribute}'.") from exc
    return factory() if callable(factory) else factory


def resolve_host(spec: str) -> AgentHost:
    """Load the host named by a `module.path:attribute` string.

    Raises rather than falling back to a default. A shell that quietly
    started with no agent because a path was misspelled would look like
    a working console attached to nothing.
    """
    host = load_spec(spec)
    if not isinstance(host, AgentHost):
        raise TypeError(f"'{spec}' does not satisfy the AgentHost protocol.")
    return host


__all__ = ["SPEC_SEPARATOR", "load_spec", "resolve_host"]
