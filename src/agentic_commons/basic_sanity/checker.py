"""Generic BasicSanity interface used by agent-specific preflight checks."""

from typing import Protocol, runtime_checkable

from agentic_commons.basic_sanity.types import BasicSanityRequest, BasicSanityResult


@runtime_checkable
class BasicSanityChecker(Protocol):
    """Deterministic checks that should run before an LLM/model call.

    The platform owns the contract; the agent owns the implementation. The
    implementation may query the agent's authoritative state, but the common
    platform never needs to know the agent's schema or domain vocabulary.
    """

    def check(self, request: BasicSanityRequest) -> BasicSanityResult: ...
