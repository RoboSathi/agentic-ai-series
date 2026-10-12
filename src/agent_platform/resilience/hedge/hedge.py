"""Hedged requests: staggered alternatives, the first good answer wins.

A slow dependency is a worse failure than a dead one: a dead one fails fast and the caller
moves on, a slow one holds the caller for its whole timeout and then usually fails anyway.
Hedging bounds that wait. The first option starts at once; when it has not finished after
`hedge_after_seconds`, or as soon as it fails, the next option starts beside it. The first
option to succeed wins and every other is cancelled. Only when every option has failed is
the last failure raised.

With `hedge_after_seconds` unset there is no staggering: an option starts only when the one
before it has failed — plain sequential fallback.

"Slow" is measured from admission (`resilience.admission`): an option still waiting for a
rate-limit slot is queued, not slow, and a backup on the same budget would only queue
beside it — so no backup starts until the newest option has been running for
`hedge_after_seconds` since it was admitted.

The options are opaque callables, so this knows nothing about models: the same pattern can
hedge a search provider or a page fetch. Each option should carry its own protection (a
`ResilientCaller`); hedging decides *when to try another*, not how to retry one.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, Protocol, TypeVar

from agent_platform.logging import get_logger
from agent_platform.resilience import admission
from agent_platform.resilience.admission import Admission

T = TypeVar("T")
# Why the next option was started: the one before was slow, or it failed.
StartReason = Literal["slow", "failed"]
Option = Callable[[], Awaitable[T]]


class HedgeListener(Protocol):
    """Told when a backup option starts and why it was started."""

    def started(self, index: int, reason: StartReason, error: BaseException | None) -> None: ...


class HedgedCall:
    """Runs options staggered by `hedge_after_seconds`; the first success wins."""

    def __init__(
        self,
        hedge_after_seconds: float | None = None,
        moves_on: Callable[[BaseException], bool] | None = None,
    ) -> None:
        """`moves_on` says whether a failure is worth trying the next option for (default all);
        a failure it rejects is raised at once and every running option is cancelled."""
        self._hedge_after = hedge_after_seconds
        self._moves_on = moves_on or (lambda error: True)
        self._logger = get_logger("resilience")

    async def run(
        self, options: Sequence[Option[T]], listener: HedgeListener | None = None
    ) -> tuple[int, T]:
        """(index of the option that answered, its result)."""
        if not options:
            raise ValueError("a hedged call needs at least one option")
        running: dict[asyncio.Task[T], int] = {}
        admissions: list[Admission] = []
        next_index = 0
        last_error: BaseException | None = None

        def start(reason: StartReason | None) -> None:
            nonlocal next_index
            index = next_index
            ready: asyncio.Future[Admission] = asyncio.get_running_loop().create_future()
            running[asyncio.ensure_future(_tracked(options[index], ready))] = index
            admissions.append(ready.result() if ready.done() else _placeholder())
            ready.add_done_callback(lambda done: admissions.__setitem__(index, done.result()))
            next_index += 1
            if reason is not None and listener is not None:
                listener.started(index, reason, last_error if reason == "failed" else None)

        start(None)
        try:
            while running:
                more = next_index < len(options)
                wait = self._slow_after(admissions[-1]) if more else None
                done, _ = await asyncio.wait(
                    running, timeout=wait, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    if self._is_slow(admissions[-1]):
                        start("slow")
                    continue
                for task in done:
                    index = running.pop(task)
                    error = task.exception()
                    if error is None:
                        return index, task.result()
                    if not self._moves_on(error):
                        raise error
                    last_error = error
                    self._logger.warning(
                        "Hedged option failed",
                        option=index,
                        error=str(error) or type(error).__name__,
                    )
                if not running and next_index < len(options):
                    start("failed")
        finally:
            await _cancel_all(running)
        assert last_error is not None
        raise last_error

    def _slow_after(self, newest: Admission) -> float | None:
        """Seconds until the newest option would count as slow (None: never staggered)."""
        if self._hedge_after is None:
            return None
        if newest.queuing:
            return self._hedge_after
        return max(0.0, self._hedge_after - newest.busy_seconds(time.monotonic()))

    def _is_slow(self, newest: Admission) -> bool:
        """Running, not queued, for at least `hedge_after_seconds` since admission."""
        if self._hedge_after is None or newest.queuing:
            return False
        return newest.busy_seconds(time.monotonic()) >= self._hedge_after


async def _tracked(option: Option[T], ready: "asyncio.Future[Admission]") -> T:
    """Run one option with its own admission record, published to the hedge."""
    ready.set_result(admission.track())
    return await option()


async def _cancel_all(running: "dict[asyncio.Task[T], int]") -> None:
    """Cancel every option still running and wait until each has unwound.

    Cancelling without waiting leaves the tasks pending: when the caller's loop then closes
    (a judge call on its own `asyncio.run`, cut short by a timeout), they are destroyed
    mid-flight and their coroutines are later resumed with no loop to run on.
    """
    for task in running:
        task.cancel()
    if running:
        await asyncio.gather(*running, return_exceptions=True)


def _placeholder() -> Admission:
    """An option not yet scheduled: started now, not queued."""
    return Admission(started_at=time.monotonic())


__all__ = ["HedgeListener", "HedgedCall", "StartReason"]
