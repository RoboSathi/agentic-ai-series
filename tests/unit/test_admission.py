"""Time queued for the rate limiter is not time spent on the call (resilience.admission)."""

import asyncio

import pytest

from agent_platform.resilience import admission
from agent_platform.resilience.hedge import HedgedCall
from agent_platform.resilience.timeout import AsyncTimeoutGuard

QUEUE, BUDGET = 0.2, 0.1


async def _queued_call() -> str:
    """Waits for a slot longer than its budget, then does a little work."""
    admission.queuing()
    await asyncio.sleep(QUEUE)
    admission.queued(QUEUE)
    await asyncio.sleep(BUDGET / 4)
    return "answered"


class TestDeadline:
    async def test_a_queued_call_is_not_timed_out_by_its_wait(self) -> None:
        assert await AsyncTimeoutGuard().run(_queued_call, seconds=BUDGET) == "answered"

    async def test_a_slow_call_still_times_out(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(BUDGET * 3)
            return "late"

        with pytest.raises(TimeoutError):
            await AsyncTimeoutGuard().run(slow, seconds=BUDGET)


class TestHedge:
    async def test_no_backup_starts_while_the_first_option_is_queued(self) -> None:
        started: list[int] = []

        class Listener:
            def started(self, index: int, reason: str, error: BaseException | None) -> None:
                started.append(index)

        async def backup() -> str:
            return "backup"

        index, answer = await HedgedCall(hedge_after_seconds=BUDGET / 2).run(
            [_queued_call, backup], Listener()
        )
        assert (index, answer) == (0, "answered") and started == []

    async def test_a_backup_starts_when_the_admitted_option_is_slow(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(BUDGET * 5)
            return "slow"

        async def backup() -> str:
            return "backup"

        assert await HedgedCall(hedge_after_seconds=BUDGET / 2).run([slow, backup]) == (
            1,
            "backup",
        )


async def test_every_enclosing_deadline_pauses_not_just_the_innermost() -> None:
    """A guard's own deadline inside a probe's: both wait out the queue."""
    inner = AsyncTimeoutGuard()
    outer = AsyncTimeoutGuard()

    async def guarded() -> str:
        return await inner.run(_queued_call, seconds=BUDGET)

    assert await outer.run(guarded, seconds=BUDGET) == "answered"
