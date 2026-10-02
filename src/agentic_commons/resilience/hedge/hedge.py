"""Hedged requests: staggered alternatives, the first good answer wins.

A slow dependency is a worse failure than a dead one: a dead one fails fast and the caller
moves on, a slow one holds the caller for its whole timeout and then usually fails anyway.
Hedging bounds that wait. The first option starts at once; when it has not finished after
`hedge_after_seconds`, or as soon as it fails, the next option starts beside it. The first
option to succeed wins and every other is cancelled. Only when every option has failed is
the last failure raised.

With `hedge_after_seconds` unset there is no staggering: an option starts only when the one
before it has failed — plain sequential fallback.

The options are opaque callables, so this knows nothing about models: the same pattern can
hedge a search provider or a page fetch. Each option should carry its own protection (a
`ResilientCaller`); hedging decides *when to try another*, not how to retry one.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, Protocol, TypeVar

from agentic_commons.logging import get_logger

T = TypeVar("T")
# Why the next option was started: the one before was slow, or it failed.
StartReason = Literal["slow", "failed"]
Option = Callable[[], Awaitable[T]]


class HedgeListener(Protocol):
    """Told when a backup option starts (and the failure that prompted it, if any), and which
    option finally answered."""

    def started(self, index: int, reason: StartReason, error: BaseException | None) -> None: ...

    def answered(self, index: int) -> None: ...


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
        next_index = 0
        last_error: BaseException | None = None

        def start(reason: StartReason | None) -> None:
            nonlocal next_index
            index = next_index
            running[asyncio.ensure_future(options[index]())] = index
            next_index += 1
            if reason is not None and listener is not None:
                listener.started(index, reason, last_error if reason == "failed" else None)

        start(None)
        try:
            while running:
                more = next_index < len(options)
                wait = self._hedge_after if more else None
                done, _ = await asyncio.wait(
                    running, timeout=wait, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    start("slow")
                    continue
                for task in done:
                    index = running.pop(task)
                    error = task.exception()
                    if error is None:
                        if listener is not None:
                            listener.answered(index)
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
            for task in running:
                task.cancel()
        assert last_error is not None
        raise last_error


__all__ = ["HedgeListener", "HedgedCall", "StartReason"]
