"""The turn bracket and the task API (ADR 0009 D1, D4; schema §6.1).

Every agent's turn runs inside `turn_trace`, which runs inside `turn_bracket`
here, so every agent gets the same turn facts without writing analytics code:

- the turn is timed, and ends `completed`, `crashed` (it raised) or `cancelled`
  (the operator stopped it), whatever ended it;
- its steps are counted at the outermost model client (`bound_step`), and the
  loop guards report its step cap;
- the grounding check reports the claims it judged;
- one `fact_turn` row is appended when it ends.

A **task** is one thing the person wants done, possibly spanning turns. An
agent opens one with `open_task(kind)` and closes it with `close_task(outcome)`;
an agent that never does still gets an **implicit task** per turn, closed with
the outcome the agent gives it or, failing that, a reserved platform outcome.
An explicit task left open carries into the session's next turn — across a
restart too, from the task facts — and is **abandoned** when the session ends.

Every outcome an agent names is declared in its overlay with `success` and
`terminal` flags; `OutcomeCatalog.require` makes an undeclared one a start-up
error rather than a silent NULL.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from agent_platform.config import get_settings
from agent_platform.config.settings import OutcomeFlags, ReservedOutcomes, TaskSettings
from agent_platform.logging import get_logger
from agent_platform.observability.facts import (
    CLOSED,
    OPENED,
    TaskEventFact,
    TurnFact,
    fact_store,
)
from agent_platform.observability.unit_context import (
    TaskBinding,
    TurnState,
    bound_turn,
    current_turn,
)

# A turn's status (UL "Turn status"): platform-defined, the same for every agent.
COMPLETED, CRASHED, CANCELLED = "completed", "crashed", "cancelled"


class UndeclaredOutcomeError(ValueError):
    """An agent named an outcome or task kind its overlay does not declare."""


class OutcomeCatalog:
    """The task kinds and outcomes declared in config, platform-reserved and the agent's."""

    def __init__(self, settings: TaskSettings | None = None) -> None:
        self._settings = settings or get_settings().analytics.tasks

    @property
    def implicit_kind(self) -> str:
        return self._settings.implicit_kind

    @property
    def reserved(self) -> ReservedOutcomes:
        """The outcomes the platform assigns: unclassified, crashed, cancelled, abandoned."""
        return self._settings.reserved

    def flags(self, outcome: str) -> OutcomeFlags:
        """The declared flags of an outcome; an undeclared one raises."""
        flags = self._settings.outcomes.get(outcome)
        if flags is None:
            raise UndeclaredOutcomeError(f"Outcome {outcome!r} is not declared in analytics.tasks")
        return flags

    def check_kind(self, kind: str) -> str:
        """`kind` itself, if declared; an undeclared one raises."""
        if kind not in self._settings.kinds:
            raise UndeclaredOutcomeError(f"Task kind {kind!r} is not declared in analytics.tasks")
        return kind

    def require(self, *, kinds: Sequence[str] = (), outcomes: Sequence[str] = ()) -> None:
        """Fail at start-up if the agent will name a kind or outcome it never declared."""
        for kind in kinds:
            self.check_kind(kind)
        for outcome in outcomes:
            self.flags(outcome)


# Explicit tasks left open, per (agent, session): the in-process half of the carry-over.
_open_tasks: dict[tuple[str, str], TaskBinding] = {}
# The screen checks each agent's composition root built, for `screen_coverage`.
_declared_screens: dict[str, list[str]] = {}


def declare_screens(agent_id: str, checks: Sequence[str]) -> None:
    """Record which screens this agent runs on a person's message.

    Its composition root calls this on every build, so a rebuilt posture (a screen
    withheld by `/harness`) replaces what an earlier build declared.
    """
    _declared_screens[agent_id] = list(checks)


@contextmanager
def turn_bracket(
    *, agent: str, session: str, turn_id: str, message: bool = True
) -> Iterator[TurnState]:
    """Bracket one turn: bind its units, time it, and append its facts however it ends.

    `message` is False for a turn that carries no message from a person (a resumed
    or aborted run), which no screen is expected to look at.
    """
    state = TurnState(turn_id=turn_id, session_id=session, task=_task_for(agent, session))
    state.screens = (_declared_screens.get(agent) or None) if message else None
    started = time.perf_counter()
    status, error = COMPLETED, ""
    with bound_turn(state):
        try:
            yield state
        except asyncio.CancelledError:
            status = CANCELLED
            raise
        except Exception as exc:
            status, error = CRASHED, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            status = CANCELLED if state.cancelled and status == COMPLETED else status
            _settle_implicit(state, status)
            _remember(agent, session, state.task)
            _append_turn(state, status, error, started)


def open_task(kind: str) -> str:
    """Open a task of a declared kind for the current turn's work; returns its id.

    Work already done in this turn under its implicit task joins the new task
    (the implicit one is never written). An explicit task still open in this
    session is abandoned: one open task per session.
    """
    turn = _require_turn()
    OutcomeCatalog().check_kind(kind)
    task_id = turn.task.task_id if turn.task.implicit else _new_task_id()
    if not turn.task.implicit and turn.task.outcome is None:
        _close(turn.task, OutcomeCatalog().reserved.abandoned)
    turn.task = TaskBinding(task_id=task_id, kind=kind, implicit=False)
    fact_store().append(TaskEventFact(task_id=turn.task.task_id, event=OPENED, kind=kind))
    return turn.task.task_id


def close_task(outcome: str) -> None:
    """Close the current task with a declared outcome (an implicit one closes at turn end)."""
    turn = _require_turn()
    OutcomeCatalog().flags(outcome)
    turn.task.outcome = outcome
    if not turn.task.implicit:
        _close(turn.task, outcome)


def current_task_kind() -> str | None:
    """The kind of the task this turn counts against (None outside a turn)."""
    turn = current_turn()
    return turn.task.kind if turn is not None else None


def mark_cancelled() -> None:
    """The operator stopped this turn (a soft stop the agent handled without raising)."""
    turn = current_turn()
    if turn is not None:
        turn.cancelled = True


def report_loop(step_cap: int, cap_hit: bool) -> None:
    """The loop guard's budget for this turn, and whether the turn ended on it."""
    turn = current_turn()
    if turn is None:
        return
    turn.step_cap = step_cap
    turn.step_cap_hit = bool(turn.step_cap_hit) or cap_hit


def report_grounding(checked: int, supported: int, unsupported: int) -> None:
    """Claims the grounding check judged in this turn (summed over its checks)."""
    turn = current_turn()
    if turn is None:
        return
    turn.claims_checked = (turn.claims_checked or 0) + checked
    turn.claims_supported = (turn.claims_supported or 0) + supported
    turn.claims_unsupported = (turn.claims_unsupported or 0) + unsupported


def end_session(agent_id: str, session_id: str) -> None:
    """The session is over: its explicit task still open, if any, is abandoned."""
    binding = _open_tasks.pop((agent_id, session_id), None)
    if binding is None:
        found = fact_store().open_task(agent_id, session_id)
        binding = TaskBinding(task_id=found[0], kind=found[1], implicit=False) if found else None
    if binding is None:
        return
    flags = OutcomeCatalog().flags(OutcomeCatalog().reserved.abandoned)
    fact_store().append(
        TaskEventFact(
            agent_id=agent_id,
            session_id=session_id,
            task_id=binding.task_id,
            event=CLOSED,
            kind=binding.kind,
            outcome=OutcomeCatalog().reserved.abandoned,
            success=flags.success,
            terminal=flags.terminal,
        )
    )
    get_logger("observability").info("Task abandoned", task_id=binding.task_id)


def _task_for(agent: str, session: str) -> TaskBinding:
    """The session's open explicit task, or a fresh implicit one for this turn."""
    binding = _open_tasks.get((agent, session))
    if binding is None:
        found = fact_store().open_task(agent, session)
        if found is not None:
            binding = TaskBinding(task_id=found[0], kind=found[1], implicit=False)
    if binding is not None:
        return binding
    kind = OutcomeCatalog().implicit_kind
    return TaskBinding(task_id=_new_task_id(), kind=kind, implicit=True)


def _settle_implicit(state: TurnState, status: str) -> None:
    """An implicit task opens and closes with its turn: the agent's outcome, else reserved."""
    task = state.task
    if not task.implicit:
        return
    reserved = OutcomeCatalog().reserved
    fallback = {COMPLETED: reserved.unclassified, CRASHED: reserved.crashed}
    outcome = task.outcome if status == COMPLETED and task.outcome else None
    outcome = outcome or fallback.get(status, reserved.cancelled)
    # Opened when the turn began, though written now: the task's duration is the turn's.
    fact_store().append(
        TaskEventFact(
            task_id=task.task_id, event=OPENED, kind=task.kind, recorded_at=state.started_at
        )
    )
    _close(task, outcome)


def _close(task: TaskBinding, outcome: str) -> None:
    flags = OutcomeCatalog().flags(outcome)
    task.outcome = outcome
    fact_store().append(
        TaskEventFact(
            task_id=task.task_id,
            event=CLOSED,
            kind=task.kind,
            outcome=outcome,
            success=flags.success,
            terminal=flags.terminal,
        )
    )


def _remember(agent: str, session: str, task: TaskBinding) -> None:
    """Carry an explicit, still-open task into the session's next turn."""
    if not task.implicit and task.outcome is None:
        _open_tasks[(agent, session)] = task
    else:
        _open_tasks.pop((agent, session), None)


def _append_turn(state: TurnState, status: str, error: str, started: float) -> None:
    fact_store().append(
        TurnFact(
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000,
            steps=state.steps,
            step_cap=state.step_cap,
            step_cap_hit=state.step_cap_hit,
            claims_checked=state.claims_checked,
            claims_supported=state.claims_supported,
            claims_unsupported=state.claims_unsupported,
            screens=state.screens,
            error=error,
        )
    )


def _require_turn() -> TurnState:
    turn = current_turn()
    if turn is None:
        raise RuntimeError("Tasks are opened and closed inside a turn (turn_trace)")
    return turn


def _new_task_id() -> str:
    return uuid.uuid4().hex


__all__ = [
    "CANCELLED",
    "COMPLETED",
    "CRASHED",
    "OutcomeCatalog",
    "UndeclaredOutcomeError",
    "close_task",
    "current_task_kind",
    "declare_screens",
    "end_session",
    "mark_cancelled",
    "open_task",
    "report_grounding",
    "report_loop",
    "turn_bracket",
]
