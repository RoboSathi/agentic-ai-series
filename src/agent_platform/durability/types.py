"""Durable-execution value types.

Thread      → one run of the graph, checkpointed node by node. Each
              customer turn is its own thread, so a turn that failed
              or is waiting on a human never blocks the next one.
Checkpoint  → the graph's state after one node finished.
Run record  → what the operator sees about a thread: how it ended,
              where it stopped, and what it is waiting on.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_platform.observability.types import utc_now

RunStatus = Literal[
    "running", "completed", "failed_resumable", "interrupted", "cancelled", "aborted"
]

# The statuses an operator can resume. A cancelled run stopped on purpose,
# part-way: its last checkpoint is where it picks up again.
RESUMABLE_STATUSES: tuple[RunStatus, ...] = ("failed_resumable", "interrupted", "cancelled")
# The statuses a stopped run can be aborted from: stopped by the operator, or failed part-way.
# An aborted run keeps its record and checkpoints (until retention) and never resumes.
ABORTABLE_STATUSES: tuple[RunStatus, ...] = ("failed_resumable", "cancelled")


class RunRecord(BaseModel):
    """One thread, as the run registry knows it."""

    thread_id: str
    agent_id: str
    session_id: str
    subject_id: str = ""
    status: RunStatus = "running"
    preview: str = ""
    next_node: str = ""
    checkpoint_id: str = ""
    error: str = ""
    attempts: int = 0
    # What an interrupted run waits for, e.g. `hitl:42`. Opaque to the
    # platform; the agent that interrupted knows what it means.
    waiting_on: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    # The trace the run's log lines carry: `grep <trace_id> logs/<agent>/*.log`.
    trace_id: str = ""


class CheckpointInfo(BaseModel):
    """One checkpoint in a thread's history."""

    checkpoint_id: str
    step: int
    source: str
    wrote: list[str] = Field(default_factory=list)
    next_nodes: list[str] = Field(default_factory=list)
    created_at: str = ""


class RunOutcome(BaseModel):
    """How one invocation of a durable run ended."""

    thread_id: str
    status: RunStatus
    values: dict[str, Any] = Field(default_factory=dict)
    interrupt: dict[str, Any] | None = None
    next_nodes: list[str] = Field(default_factory=list)
    checkpoint_id: str = ""
    checkpoints: int = 0


class ResumableRunError(RuntimeError):
    """A run stopped part-way; its state is checkpointed and it can resume."""

    def __init__(self, thread_id: str, node: str, error: str) -> None:
        super().__init__(f"'{node}' failed: {error} — state is checkpointed as thread {thread_id}")
        self.thread_id = thread_id
        self.node = node
        self.error = error


class ResumeResult(BaseModel):
    """What resuming a thread produced, for the console to show."""

    thread_id: str
    session_id: str
    status: RunStatus
    response: str = ""
