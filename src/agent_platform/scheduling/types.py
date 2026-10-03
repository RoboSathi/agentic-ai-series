"""Task-graph value types.

A *task graph* is a set of units of work with dependencies between them.
A *wave* is every node whose dependencies are all satisfied at one moment:
the set that may run concurrently now. Nothing here knows what a node
does — the caller supplies a worker; this package decides only *when*.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OutcomeStatus = Literal["done", "failed", "blocked", "skipped"]


class TaskNode(BaseModel):
    """One unit of work and the nodes it waits for."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    depends_on: tuple[str, ...] = ()


class TaskOutcome(BaseModel):
    """How one node ended.

    `blocked` means a dependency failed or was itself blocked, so the node
    never ran; `skipped` means the caller stopped scheduling (a budget, a
    cancellation) before the node's turn came.
    """

    node_id: str
    status: OutcomeStatus
    value: Any = None
    error: str = ""
    wave: int = 0


class InvalidTaskGraphError(ValueError):
    """The graph cannot be scheduled: a cycle, a duplicate, or an unknown dependency."""

    def __init__(self, message: str, cycle: list[str] | None = None) -> None:
        super().__init__(message)
        self.cycle = cycle or []


class WaveSummary(BaseModel):
    """What one wave ran and how it went — for a caller's progress view."""

    wave: int
    node_ids: list[str] = Field(default_factory=list)
    done: int = 0
    failed: int = 0
