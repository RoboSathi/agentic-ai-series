"""Dependency-aware concurrent scheduling (a generic mechanism).

    graph = TaskGraph([TaskNode(node_id="a"), TaskNode(node_id="b", depends_on=("a",))])
    graph.ready({})          # ["a"]: what may run now
    async with AsyncDagScheduler(concurrency=4).slot(): ...

A node's failure blocks its dependents and nothing else (`TaskGraph.blocked`).
"""

from agent_platform.runtime.scheduling.graph import TaskGraph, find_cycle
from agent_platform.runtime.scheduling.scheduler import AsyncDagScheduler
from agent_platform.runtime.scheduling.types import (
    InvalidTaskGraphError,
    OutcomeStatus,
    TaskNode,
)

__all__ = [
    "AsyncDagScheduler",
    "InvalidTaskGraphError",
    "OutcomeStatus",
    "TaskGraph",
    "TaskNode",
    "find_cycle",
]
