"""Time spent queued for the rate limiter is not time spent on the call.

The rate limiter waits at the transport, deep inside a call that is already timed:
a per-attempt deadline (`AsyncTimeoutGuard`) and a hedge's "is it slow?" clock both
start when the call starts. Counting the queue against them turns throttling into a
loop — a request queued past its 8 s deadline times out, is retried, queues again and
times out again; a model queued past `hedge_after_seconds` looks slow, so a backup on
the same budget starts and queues beside it. Every lap spends more of the budget the
queue is waiting on.

So a call's clocks run from *admission*. The transport reports each wait here: while a
call is `queuing` its deadline is paused (the budget it had left is kept), and once it is
`queued` (admitted) the deadline resumes with that budget. The hedge reads an option's
`Admission` to tell "slow" from "still queued".
"""

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class _Paused:
    """A deadline set aside while its call waits for a slot."""

    scope: asyncio.Timeout
    remaining: float


@dataclass
class Admission:
    """One call's standing with the rate limiter."""

    started_at: float
    # Waiting for a slot right now: the call is queued, not slow.
    queuing: bool = False
    queued_seconds: float = 0.0
    admitted_at: float | None = None

    def busy_seconds(self, now: float) -> float:
        """How long the call has been running since it was last admitted (or started)."""
        return now - (self.admitted_at if self.admitted_at is not None else self.started_at)


_ADMISSION: ContextVar[Admission | None] = ContextVar("admission", default=None)
# Every deadline the current call runs under, outermost first: a guard's own 8 s deadline
# inside a probe's 20 s one. A wait for a slot pauses them all, not just the innermost.
_DEADLINES: ContextVar[tuple[asyncio.Timeout, ...]] = ContextVar("deadlines", default=())
_PAUSED: ContextVar[tuple[_Paused, ...]] = ContextVar("paused_deadlines", default=())


def track() -> Admission:
    """Start tracking the current task's call (a hedge option, say)."""
    admission = Admission(started_at=time.monotonic())
    _ADMISSION.set(admission)
    return admission


@contextmanager
def deadline_scope(scope: asyncio.Timeout) -> Iterator[None]:
    """Make `scope` the deadline a rate-limit wait extends, for the current task."""
    token = _DEADLINES.set((*_DEADLINES.get(), scope))
    try:
        yield
    finally:
        _DEADLINES.reset(token)


def queuing() -> None:
    """The call is about to wait for a slot: pause its deadline."""
    admission = _ADMISSION.get()
    if admission is not None:
        admission.queuing = True
    now = asyncio.get_running_loop().time()
    paused: list[_Paused] = []
    for scope in _DEADLINES.get():
        when = scope.when()
        if when is not None:
            paused.append(_Paused(scope, max(0.0, when - now)))
            scope.reschedule(None)
    _PAUSED.set(tuple(paused))


def queued(seconds: float) -> None:
    """The call was admitted after `seconds` in the queue: resume its deadline."""
    admission = _ADMISSION.get()
    if admission is not None:
        admission.queuing = False
        admission.queued_seconds += seconds
        admission.admitted_at = time.monotonic()
    now = asyncio.get_running_loop().time()
    for paused in _PAUSED.get():
        paused.scope.reschedule(now + paused.remaining)
    _PAUSED.set(())
