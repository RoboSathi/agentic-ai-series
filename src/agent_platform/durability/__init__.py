"""Durable execution: checkpointed runs that survive failure (LLD §13).

    checkpointer = create_checkpointer(state_types=model_types(MyState))
    graph = builder.compile(checkpointer=checkpointer)
    runner = DurableGraphRunner(graph, checkpointer, RunRegistry(store))
    await runner.start(thread_id, initial_state, ...)   # may raise ResumableRunError
    await runner.resume(thread_id)                        # re-runs only the failed node

An interrupted run (a graph that called `interrupt({"waiting_on": ...})`)
resumes with the decision it was waiting for: `runner.resume(thread, decision)`.
"""

from agent_platform.durability.checkpointer import (
    CheckpointKeyError,
    SqliteCheckpointer,
    create_checkpointer,
    model_types,
)
from agent_platform.durability.protocols import DurableAgent
from agent_platform.durability.registry import ORPHANED_REASON, RunRegistry
from agent_platform.durability.runner import (
    SUMMARY,
    WAITING_ON,
    DurableGraphRunner,
    ResumeRefusedError,
    purge_expired,
)
from agent_platform.durability.types import (
    RESUMABLE_STATUSES,
    CheckpointInfo,
    ResumableRunError,
    ResumeResult,
    RunOutcome,
    RunRecord,
    RunStatus,
)

__all__ = [
    "ORPHANED_REASON",
    "RESUMABLE_STATUSES",
    "SUMMARY",
    "WAITING_ON",
    "CheckpointInfo",
    "CheckpointKeyError",
    "DurableAgent",
    "DurableGraphRunner",
    "ResumableRunError",
    "ResumeRefusedError",
    "ResumeResult",
    "RunOutcome",
    "RunRecord",
    "RunRegistry",
    "RunStatus",
    "SqliteCheckpointer",
    "create_checkpointer",
    "model_types",
    "purge_expired",
]
