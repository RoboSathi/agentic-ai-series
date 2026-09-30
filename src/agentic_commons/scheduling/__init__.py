"""Dependency-aware concurrent scheduling (a generic mechanism).

    graph = TaskGraph([TaskNode(node_id="a"), TaskNode(node_id="b", depends_on=("a",))])
    outcomes = await AsyncDagScheduler(concurrency=4).run(graph, worker)

A node's failure blocks its dependents and nothing else.
"""

from agentic_commons.scheduling.graph import TaskGraph, find_cycle
from agentic_commons.scheduling.scheduler import AsyncDagScheduler, WaveObserver, Worker
from agentic_commons.scheduling.types import (
    InvalidTaskGraphError,
    OutcomeStatus,
    TaskNode,
    TaskOutcome,
    WaveSummary,
)

__all__ = [
    "AsyncDagScheduler",
    "InvalidTaskGraphError",
    "OutcomeStatus",
    "TaskGraph",
    "TaskNode",
    "TaskOutcome",
    "WaveObserver",
    "WaveSummary",
    "Worker",
    "find_cycle",
]
