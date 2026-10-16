"""Task-graph value types.

A *task graph* is a set of units of work with dependencies between them.
A *batch* is every node whose dependencies are all satisfied at one moment:
the set that may run concurrently now. Nothing here knows what a node
does — the caller supplies a worker; this package decides only *when*.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

OutcomeStatus = Literal["done", "failed", "blocked", "skipped"]


class TaskNode(BaseModel):
    """One unit of work and the nodes it waits for."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    depends_on: tuple[str, ...] = ()


class InvalidTaskGraphError(ValueError):
    """The graph cannot be scheduled: a cycle, a duplicate, or an unknown dependency."""

    def __init__(self, message: str, cycle: list[str] | None = None) -> None:
        super().__init__(message)
        self.cycle = cycle or []
