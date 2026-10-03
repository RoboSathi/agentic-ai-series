"""Execution isolation for tool calls — PARKED (LLD §14).

Interface-only this phase. The folder and contracts exist so the module
boundary is real and nothing downstream has to be restructured when the
implementation lands; no code calls this package yet.

Deliberately scoped down. E34 documents a five-tier isolation ladder
(process policy → OS sandbox → container → microVM → remote worker), but
only the container tier is planned for implementation, because the
others need a deployment target this repo does not have. Shipping three
empty tiers would suggest isolation that is not there.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from agent_platform.tools.types import ToolCall

IsolationTier = Literal["process_policy", "docker_container"]


class SandboxResult(BaseModel):
    """What came back from an isolated execution."""

    succeeded: bool
    output: str = ""
    error: str | None = None
    tier: IsolationTier = "process_policy"
    duration_ms: float = 0.0


@runtime_checkable
class ExecutionSandbox(Protocol):
    """Runs a tool call inside an isolation boundary."""

    async def run(self, call: ToolCall, isolation_tier: IsolationTier) -> SandboxResult: ...


@runtime_checkable
class IsolationTierSelector(Protocol):
    """Chooses how much isolation a call needs, from its blast radius."""

    def tier_for(self, call: ToolCall) -> IsolationTier: ...


__all__ = [
    "ExecutionSandbox",
    "IsolationTier",
    "IsolationTierSelector",
    "SandboxResult",
]
