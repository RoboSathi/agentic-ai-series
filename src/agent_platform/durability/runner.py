"""Running a compiled graph durably: start, fail resumably, resume.

The runner owns the failure taxonomy so no agent re-derives it:

    node raised            → `failed_resumable`: the last checkpoint holds
                             every node that finished; `next` is the one
                             that did not. resuming re-runs only that node.
    graph called interrupt → `interrupted`: waiting on something outside
                             the process (a human). The interrupt payload's
                             `waiting_on` names it, so the decision can find
                             its way back to this exact thread.
    graph reached the end  → `completed`.

A re-run node must be safe to run twice. Side-effecting tools already
are — the executor replays them by an idempotency key built from the
action and its arguments — so a refund that committed before the crash
is replayed, not paid again. A model call is simply made again; the
resume card says so, because that is a real cost.
"""

import asyncio
from datetime import timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agent_platform.config import get_settings
from agent_platform.config.settings import DurabilitySettings
from agent_platform.durability.checkpointer import SqliteCheckpointer
from agent_platform.durability.registry import RunRegistry
from agent_platform.durability.types import (
    ABORTABLE_STATUSES,
    RESUMABLE_STATUSES,
    CheckpointInfo,
    ResumableRunError,
    RunOutcome,
)
from agent_platform.logging import get_current_trace_id, get_logger
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.types import utc_now

# The key in an interrupt payload that says what the run is waiting on.
WAITING_ON = "waiting_on"
# The key in an interrupt payload holding the one line a person should read.
SUMMARY = "summary"
# Payload keys that are for the registry, not for the card.
_NOT_SHOWN = frozenset({WAITING_ON, SUMMARY})


class ResumeRefusedError(RuntimeError):
    """The thread does not exist or is not in a resumable state."""


class DurableGraphRunner:
    """Runs one compiled graph against a checkpointer and a run registry."""

    def __init__(
        self,
        graph: Any,
        checkpointer: SqliteCheckpointer,
        registry: RunRegistry,
        settings: DurabilitySettings | None = None,
    ) -> None:
        self._graph = graph
        self._checkpointer = checkpointer
        self._registry = registry
        self._settings = settings or get_settings().durability
        self._logger = get_logger("persistence")
        # The graph invocation in flight per thread, and the threads an
        # operator asked to stop: only a cancellation asked for here is a
        # clean stop. Any other (process shutdown) still propagates.
        self._active: dict[str, asyncio.Future[Any]] = {}
        self._cancel_requested: set[str] = set()

    @property
    def registry(self) -> RunRegistry:
        return self._registry

    async def start(
        self,
        thread_id: str,
        payload: Any,
        *,
        session_id: str,
        subject_id: str,
        preview: str,
        reporter: StageReporter | None = None,
    ) -> RunOutcome:
        """Run a new thread from its initial state."""
        self._registry.start(
            thread_id,
            session_id,
            subject_id,
            preview[: self._settings.preview_chars],
            trace_id=get_current_trace_id() or "",
        )
        return await self._invoke(thread_id, payload, reporter)

    async def resume(
        self,
        thread_id: str,
        decision: Any | None = None,
        reporter: StageReporter | None = None,
    ) -> RunOutcome:
        """Continue a failed or interrupted thread from its last checkpoint."""
        record = self._registry.get(thread_id)
        if record is None or record.status not in RESUMABLE_STATUSES:
            state = record.status if record else "unknown"
            raise ResumeRefusedError(f"Thread {thread_id} is {state}; nothing to resume")
        if record.status == "interrupted" and decision is None:
            raise ResumeRefusedError(
                f"Thread {thread_id} is waiting on {record.waiting_on}; it resumes with a decision"
            )
        self._registry.attempt(thread_id)
        self._logger.info(
            "Resuming run",
            thread=thread_id,
            checkpoint=record.checkpoint_id,
            node=record.next_node,
            attempt=record.attempts + 1,
            resumed_from_trace=record.trace_id,
        )
        if reporter is not None:
            reporter.completed(
                "run_resumed",
                event_kind="system",
                summary=_resume_summary(record.status, record.next_node),
                fields={
                    "why": _RESUME_WHY[record.status],
                    "continues_at": record.next_node or "—",
                    "attempt": record.attempts + 1,
                    "previous_error": record.error or None,
                },
            )
        payload: Command[Any] | None = (
            Command(resume=decision) if record.status == "interrupted" else None
        )
        return await self._invoke(thread_id, payload, reporter)

    async def abort(self, thread_id: str, reporter: StageReporter | None = None) -> RunOutcome:
        """End a stopped run for good: it keeps its record and checkpoints, and never resumes.

        Nothing is undone here: a Stop already left the run's effects whole (a saga it cut
        off compensated), so aborting is only the decision not to go on.
        """
        record = self._registry.get(thread_id)
        if record is None or record.status not in ABORTABLE_STATUSES:
            state = record.status if record else "unknown"
            raise ResumeRefusedError(f"Thread {thread_id} is {state}; nothing to abort")
        snapshot = await self._graph.aget_state(_config(thread_id))
        self._registry.mark(
            thread_id,
            "aborted",
            next_node=record.next_node,
            checkpoint_id=record.checkpoint_id,
            error=self._settings.abort_reason,
        )
        self._logger.warning("Run aborted; record kept", thread=thread_id, node=record.next_node)
        if reporter is not None:
            reporter.warning(
                "run_aborted",
                event_kind="system",
                summary=self._settings.stop_report.aborted,
                fields={"stopped_at": record.next_node or "—", "thread": thread_id},
            )
        return RunOutcome(
            thread_id=thread_id,
            status="aborted",
            values=dict(snapshot.values or {}),
            next_nodes=[node for node in record.next_node.split(",") if node],
            checkpoint_id=record.checkpoint_id,
            checkpoints=self._count(thread_id),
        )

    async def values(self, thread_id: str) -> dict[str, Any]:
        """The latest state of a thread (empty if it has none)."""
        snapshot = await self._graph.aget_state(_config(thread_id))
        return dict(snapshot.values or {})

    async def history(self, thread_id: str, limit: int) -> list[CheckpointInfo]:
        """A thread's checkpoints, newest first, each with the node that wrote it.

        A snapshot records what runs *next*; the node that produced a
        checkpoint is therefore the `next` of the one before it.
        """
        snapshots = [
            snapshot
            async for snapshot in self._graph.aget_state_history(_config(thread_id), limit=limit)
        ]
        return [
            _info(snapshot, snapshots[index + 1] if index + 1 < len(snapshots) else None)
            for index, snapshot in enumerate(snapshots)
        ]

    def purge_expired(self) -> int:
        """Delete completed threads older than the retention window."""
        return purge_expired(self._checkpointer, self._registry, self._settings)

    def cancel(self, thread_id: str) -> bool:
        """Stop a running thread at once; it keeps every checkpoint it wrote.

        False when the thread is not running in this process. The node in
        flight is abandoned and runs again on resume — a side effect it had
        already committed replays from its idempotency record.
        """
        running = self._active.get(thread_id)
        if running is None or running.done():
            return False
        self._cancel_requested.add(thread_id)
        running.cancel()
        return True

    async def _invoke(
        self, thread_id: str, payload: Any, reporter: StageReporter | None
    ) -> RunOutcome:
        config = _config(thread_id)
        running = asyncio.ensure_future(
            self._graph.ainvoke(payload, config, durability=self._settings.mode)
        )
        self._active[thread_id] = running
        try:
            await running
        except asyncio.CancelledError:
            outcome = await self._cancelled(thread_id, reporter)
            if thread_id not in self._cancel_requested:
                raise
            return outcome
        except Exception as exc:
            raise await self._fail(thread_id, exc, reporter) from exc
        finally:
            self._active.pop(thread_id, None)
            self._cancel_requested.discard(thread_id)
        return await self._settle(thread_id, reporter)

    async def _cancelled(self, thread_id: str, reporter: StageReporter | None) -> RunOutcome:
        """Record a stopped run as resumable from its last completed node."""
        snapshot = await self._graph.aget_state(_config(thread_id))
        outcome = RunOutcome(
            thread_id=thread_id,
            status="cancelled",
            values=dict(snapshot.values or {}),
            next_nodes=list(snapshot.next),
            checkpoint_id=str(snapshot.config["configurable"].get("checkpoint_id", "")),
            checkpoints=self._count(thread_id),
        )
        node = outcome.next_nodes[0] if outcome.next_nodes else ""
        self._registry.mark(
            thread_id,
            "cancelled",
            next_node=",".join(outcome.next_nodes),
            checkpoint_id=outcome.checkpoint_id,
            error=self._settings.cancel_reason,
        )
        self._logger.warning("Run cancelled; state checkpointed", thread=thread_id, node=node)
        if reporter is not None:
            reporter.warning(
                "run_cancelled",
                event_kind="system",
                summary=f"Cancelled before {node or 'the end'} — resumable",
                fields={
                    "continues_at": node or "—",
                    "checkpoints": outcome.checkpoints,
                    "state": "saved — nothing already done will be repeated",
                    "thread": thread_id,
                },
            )
        return outcome

    async def _settle(self, thread_id: str, reporter: StageReporter | None) -> RunOutcome:
        """Read where the run stopped and record it."""
        snapshot = await self._graph.aget_state(_config(thread_id))
        checkpoint_id = str(snapshot.config["configurable"].get("checkpoint_id", ""))
        interrupt = _first_interrupt(snapshot)
        outcome = RunOutcome(
            thread_id=thread_id,
            status="interrupted" if interrupt is not None else "completed",
            values=dict(snapshot.values or {}),
            interrupt=interrupt,
            next_nodes=list(snapshot.next),
            checkpoint_id=checkpoint_id,
            checkpoints=self._count(thread_id),
        )
        waiting_on = str((interrupt or {}).get(WAITING_ON, ""))
        self._registry.mark(
            thread_id,
            outcome.status,
            next_node=",".join(outcome.next_nodes),
            checkpoint_id=checkpoint_id,
            waiting_on=waiting_on,
        )
        if interrupt is not None:
            self._logger.info(
                "Run paused", thread=thread_id, checkpoint=checkpoint_id, waiting_on=waiting_on
            )
        if interrupt is not None and reporter is not None:
            reporter.waiting(
                "run_interrupted",
                event_kind="system",
                summary=str(interrupt.get(SUMMARY) or f"Waiting on {waiting_on or 'a decision'}"),
                fields={k: v for k, v in interrupt.items() if k not in _NOT_SHOWN},
            )
        return outcome

    async def _fail(
        self, thread_id: str, exc: Exception, reporter: StageReporter | None
    ) -> ResumableRunError:
        """Record a resumable failure and tell the operator how to resume it."""
        snapshot = await self._graph.aget_state(_config(thread_id))
        node = snapshot.next[0] if snapshot.next else ""
        checkpoint_id = str(snapshot.config["configurable"].get("checkpoint_id", ""))
        error = str(exc) or type(exc).__name__
        self._registry.mark(
            thread_id,
            "failed_resumable",
            next_node=node,
            checkpoint_id=checkpoint_id,
            error=error,
        )
        self._logger.error(
            "Run failed; state checkpointed", thread=thread_id, node=node, error=error
        )
        if reporter is not None:
            reporter.failed(
                "run_failed_resumable",
                event_kind="system",
                summary=f"{node or 'run'} failed — resumable",
                fields={
                    "failed_at": node or "—",
                    "error": error,
                    "state": "saved — nothing already done will be repeated",
                    "thread": thread_id,
                },
            )
        return ResumableRunError(thread_id, node, error)

    def _count(self, thread_id: str) -> int:
        return len(list(self._checkpointer.list(_config(thread_id))))


def purge_expired(
    checkpointer: SqliteCheckpointer,
    registry: RunRegistry,
    settings: DurabilitySettings | None = None,
) -> int:
    """Delete completed threads older than the retention window.

    Checkpoints hold the conversation; keeping them forever is keeping
    customers' words forever. Interrupted and failed runs are kept: they
    are work still owed, whatever their age.
    """
    resolved = settings or get_settings().durability
    expired = registry.expired(utc_now() - timedelta(days=resolved.retention_days))
    for thread_id in expired:
        checkpointer.delete_thread(thread_id)
        registry.forget(thread_id)
    if expired:
        get_logger("persistence").info("Expired threads purged", count=len(expired))
    return len(expired)


# Why a run is being resumed, by how it stopped.
_RESUME_WHY = {
    "interrupted": "a decision arrived",
    "failed_resumable": "the last attempt failed",
    "cancelled": "the operator cancelled it",
}


def _resume_summary(status: str, next_node: str) -> str:
    """The resume card's one line, by how the run stopped."""
    step = next_node or "the last step"
    if status == "interrupted":
        return "Decision received — continuing the paused turn"
    if status == "cancelled":
        return f"Resuming after cancel from {step}"
    return f"Retrying from {step}"


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _first_interrupt(snapshot: Any) -> dict[str, Any] | None:
    interrupts = getattr(snapshot, "interrupts", ()) or ()
    if not interrupts:
        return None
    value = interrupts[0].value
    return dict(value) if isinstance(value, dict) else {"value": value}


def _info(snapshot: Any, previous: Any | None) -> CheckpointInfo:
    metadata = snapshot.metadata or {}
    return CheckpointInfo(
        checkpoint_id=str(snapshot.config["configurable"].get("checkpoint_id", "")),
        step=int(metadata.get("step", -1)),
        source=str(metadata.get("source", "")),
        wrote=list(previous.next) if previous is not None else [],
        next_nodes=list(snapshot.next),
        created_at=str(snapshot.created_at or ""),
    )
