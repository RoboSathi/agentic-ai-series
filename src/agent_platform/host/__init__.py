"""The shell↔agent contract: what the operator surface needs from an agent."""

from agent_platform.host.loader import SPEC_SEPARATOR, load_spec, resolve_host
from agent_platform.host.protocols import (
    ActionHost,
    AgentHost,
    CancellableHost,
    GreetingHost,
    HostAction,
    StatusHost,
)

__all__ = [
    "SPEC_SEPARATOR",
    "ActionHost",
    "AgentHost",
    "CancellableHost",
    "GreetingHost",
    "HostAction",
    "StatusHost",
    "load_spec",
    "resolve_host",
]
