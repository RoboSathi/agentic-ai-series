"""A Stop mid-saga undoes what committed, the same as a failure does.

The operator's cancel arrives as `CancelledError`, which is not an `Exception`. Caught only as
one, a refund stopped between "mark refunded" and "record payment" stayed half-applied.
"""

import asyncio

import pytest

from agent_platform.resilience import CompensatingSaga
from agent_platform.resilience.types import SagaStep


class _Ledger:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def step(self, name: str) -> SagaStep:
        async def act() -> None:
            self.log.append(f"do {name}")

        async def undo() -> None:
            self.log.append(f"undo {name}")

        return SagaStep(name=name, action=act, compensation=undo)

    def held(self, name: str) -> SagaStep:
        """A step that waits — where the operator presses Stop."""

        async def act() -> None:
            self.started.set()
            await self.release.wait()
            self.log.append(f"do {name}")

        return SagaStep(name=name, action=act, compensation=None)


async def test_a_stop_between_steps_compensates_what_committed_and_stops() -> None:
    ledger = _Ledger()
    steps = [ledger.step("mark_refunded"), ledger.held("record_payment")]
    task = asyncio.ensure_future(CompensatingSaga().run(steps))
    await ledger.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ledger.log == ["do mark_refunded", "undo mark_refunded"]


async def test_a_second_stop_cannot_cut_the_compensation_short() -> None:
    ledger = _Ledger()
    undo_started, undo_release = asyncio.Event(), asyncio.Event()

    async def slow_undo() -> None:
        undo_started.set()
        await undo_release.wait()
        ledger.log.append("undo mark_refunded")

    first = SagaStep(
        name="mark_refunded", action=ledger.step("mark_refunded").action, compensation=slow_undo
    )
    task = asyncio.ensure_future(CompensatingSaga().run([first, ledger.held("record_payment")]))
    await ledger.started.wait()
    task.cancel()
    await undo_started.wait()
    task.cancel()
    undo_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert "undo mark_refunded" in ledger.log
