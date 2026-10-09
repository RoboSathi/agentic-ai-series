"""The units of work a fact belongs to: the turn, its task and its step (ADR 0009 D1, D5).

Pure context, no I/O, so both the fact store (which stamps every row from it)
and the turn bracket (`units.py`, which writes the turn and task facts) can
import it without importing each other.

A turn binds one `TurnState`; a step binds one `StepState` inside it. Both are
mutable objects held by a context variable, so concurrent work inside a turn —
a hedged backup model, a parallel screen — counts against the same turn and the
same step rather than a copy of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class TaskBinding:
    """The task a turn's work counts against."""

    task_id: str
    kind: str
    implicit: bool
    # Set by `close_task` inside the turn; the bracket writes the close.
    outcome: str | None = None


@dataclass
class Generation:
    """One completion's text and its tokens with their log-probabilities (plain values)."""

    model: str
    content: str
    tokens: list[tuple[str, float]]


@dataclass
class TurnState:
    """What the bracket learns about one turn while it runs."""

    turn_id: str
    session_id: str
    task: TaskBinding
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    steps: int = 0
    cancelled: bool = False
    step_cap: int | None = None
    step_cap_hit: bool | None = None
    claims_checked: int | None = None
    claims_supported: int | None = None
    claims_unsupported: int | None = None
    screens: list[str] | None = None
    # Every completion this turn that came with token log-probabilities, oldest first:
    # what the confidence check reads to score a reply's sentences.
    generations: list[Generation] = field(default_factory=list)


def remember_generation(generation: Generation) -> None:
    """Keep a completion's token confidences on the turn (nothing outside a turn)."""
    turn = _turn.get()
    if turn is not None:
        turn.generations.append(generation)


@dataclass
class StepState:
    """One logical model call: its number in the turn, and how many attempts it took."""

    number: int | None
    attempts: int = 0


_turn: ContextVar[TurnState | None] = ContextVar("analytics_turn", default=None)
_step: ContextVar[StepState | None] = ContextVar("analytics_step", default=None)


def current_turn() -> TurnState | None:
    """The turn running in this context, or None outside one (a background job)."""
    return _turn.get()


@contextmanager
def bound_turn(state: TurnState) -> Iterator[TurnState]:
    """Bind a turn for everything inside the block; reset on exit, never leaked."""
    token = _turn.set(state)
    try:
        yield state
    finally:
        _turn.reset(token)


@contextmanager
def bound_step() -> Iterator[StepState]:
    """One step: numbered within the bound turn (None outside a turn)."""
    turn = _turn.get()
    number: int | None = None
    if turn is not None:
        turn.steps += 1
        number = turn.steps
    state = StepState(number=number)
    token = _step.set(state)
    try:
        yield state
    finally:
        _step.reset(token)


def next_attempt() -> int:
    """The attempt number of the provider call about to be made in the current step."""
    step = _step.get()
    if step is None:
        return 1
    step.attempts += 1
    return step.attempts


def unit_fields() -> dict[str, object]:
    """The unit keys a fact is stamped with: turn, task and step (blank outside a turn)."""
    turn = _turn.get()
    step = _step.get()
    # Outside a model call, work belongs to the step that led to it (the tools the model
    # just proposed); before the first step (a screen on the message) there is none.
    latest = turn.steps or None if turn else None
    return {
        "turn_id": turn.turn_id if turn else "",
        "task_id": turn.task.task_id if turn else "",
        "step": step.number if step else latest,
    }


__all__ = [
    "Generation",
    "StepState",
    "remember_generation",
    "TaskBinding",
    "TurnState",
    "bound_step",
    "bound_turn",
    "current_turn",
    "next_attempt",
    "unit_fields",
]
