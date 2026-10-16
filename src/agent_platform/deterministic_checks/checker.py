"""Generic DeterministicCheck interface used by agent-specific preflight checks."""

from typing import Any, Protocol, runtime_checkable

from agent_platform.deterministic_checks.types import (
    DeterministicCheckRequest,
    DeterministicCheckResult,
)
from agent_platform.security.pii.active import tokenise_tree_if_on


@runtime_checkable
class DeterministicChecker(Protocol):
    """Deterministic checks that should run before an LLM/model call.

    The platform owns the contract; the agent owns the implementation. The
    implementation may query the agent's authoritative state, but the common
    platform never needs to know the agent's schema or domain vocabulary.
    """

    def check(self, request: DeterministicCheckRequest) -> DeterministicCheckResult: ...


class AdmittedChecker:
    """A checker whose findings and facts enter the harness as tokens (`security.pii.source`).

    A check reads the agent's own records — an order's delivery address, a carrier's note —
    and its facts are kept in the turn's state, checkpointed and shown. So they are tokenised
    where they come out, like a tool's result, whatever checker an agent injects.
    """

    def __init__(self, inner: DeterministicChecker) -> None:
        self._inner = inner

    @property
    def inner(self) -> DeterministicChecker:
        """The agent's own checker, as it was injected."""
        return self._inner

    def check(self, request: DeterministicCheckRequest) -> DeterministicCheckResult:
        result = self._inner.check(request)
        return DeterministicCheckResult.model_validate(
            tokenise_tree_if_on(result.model_dump(mode="json"))
        )

    def __getattr__(self, name: str) -> Any:
        """Anything else read from a checker is the wrapped checker's."""
        return getattr(self._inner, name)
