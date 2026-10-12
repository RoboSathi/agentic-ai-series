"""Dependency-aware concurrent scheduling (a generic mechanism).

    graph = TaskGraph([TaskNode(node_id="a"), TaskNode(node_id="b", depends_on=("a",))])
    outcomes = await AsyncDagScheduler(concurrency=4).run(graph, worker)

A node's failure blocks its dependents and nothing else.
"""

from agent_platform.runtime.scheduling.graph import TaskGraph, find_cycle
from agent_platform.runtime.scheduling.scheduler import AsyncDagScheduler, BatchObserver, Worker
from agent_platform.runtime.scheduling.types import (
    BatchSummary,
    InvalidTaskGraphError,
    OutcomeStatus,
    TaskNode,
    TaskOutcome,
)

__all__ = [
    "AsyncDagScheduler",
    "InvalidTaskGraphError",
    "OutcomeStatus",
    "TaskGraph",
    "TaskNode",
    "TaskOutcome",
    "BatchObserver",
    "BatchSummary",
    "Worker",
    "find_cycle",
]
