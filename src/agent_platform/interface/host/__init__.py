"""The shell↔agent contract: what the operator surface needs from an agent."""

from agent_platform.interface.host.loader import SPEC_SEPARATOR, load_spec, resolve_host
from agent_platform.interface.host.protocols import (
    ActionHost,
    AgentHost,
    CancellableHost,
    GreetingHost,
    HostAction,
    HousekeepingHost,
    MemoryHost,
    StatusHost,
    SubjectHost,
)

__all__ = [
    "SPEC_SEPARATOR",
    "ActionHost",
    "AgentHost",
    "CancellableHost",
    "GreetingHost",
    "HostAction",
    "HousekeepingHost",
    "MemoryHost",
    "StatusHost",
    "SubjectHost",
    "load_spec",
    "resolve_host",
]
