"""Budget guard: reserve before await, commit/release, exhaustion events."""

import asyncio

import pytest

from agent_platform.loop_control import SECONDS, BudgetExhaustedError, BudgetGuard
from agent_platform.loop_control.types import LoopControlReport


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestReserveCommitRelease:
    def test_reservation_holds_capacity_until_released(self) -> None:
        guard = BudgetGuard({"searches": 2})
        first = guard.reserve("searches")
        assert guard.remaining("searches") == 1
        guard.release(first)
        assert guard.remaining("searches") == 2

    def test_commit_can_record_the_actual_amount(self) -> None:
        guard = BudgetGuard({"tokens": 100})
        held = guard.reserve("tokens", 50)
        guard.commit(held, actual=20)
        assert guard.remaining("tokens") == 80
        assert guard.snapshot()["tokens"].spent == 20

    def test_reserve_raises_at_the_cap(self) -> None:
        guard = BudgetGuard({"fetches": 1})
        guard.reserve("fetches")
        with pytest.raises(BudgetExhaustedError):
            guard.reserve("fetches")

    def test_try_reserve_returns_none_at_the_cap(self) -> None:
        guard = BudgetGuard({"fetches": 0})
        assert guard.try_reserve("fetches") is None

    def test_committing_twice_is_harmless(self) -> None:
        guard = BudgetGuard({"searches": 3})
        held = guard.reserve("searches")
        guard.commit(held)
        guard.commit(held)
        assert guard.remaining("searches") == 2

    def test_unknown_counter_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            BudgetGuard({}).reserve("nope")


class TestNoOvershootUnderParallelWorkers:
    async def test_parallel_workers_never_spend_more_than_the_cap(self) -> None:
        guard = BudgetGuard({"searches": 3})
        spent = 0

        async def worker() -> None:
            nonlocal spent
            held = guard.try_reserve("searches")
            if held is None:
                return
            await asyncio.sleep(0)
            spent += 1
            guard.commit(held)

        await asyncio.gather(*(worker() for _ in range(10)))
        assert spent == 3


class TestExhaustion:
    def test_the_event_fires_once_per_counter(self) -> None:
        events: list[LoopControlReport] = []
        guard = BudgetGuard({"searches": 1}, on_exhausted=events.append)
        guard.commit(guard.reserve("searches"))
        guard.try_reserve("searches")
        guard.try_reserve("searches")
        assert len(events) == 1
        assert events[0].decision == "abort"
        assert events[0].checks[0].reason_code == "BUDGET_EXHAUSTED"

    def test_charge_reports_exhaustion(self) -> None:
        guard = BudgetGuard({"tokens": 10})
        assert guard.charge("tokens", 4) is True
        assert guard.charge("tokens", 6) is False
        assert guard.exhausted_counters() == ["tokens"]

    def test_wall_clock_expiry_refuses_reservations(self) -> None:
        clock = FakeClock()
        guard = BudgetGuard({"searches": 5}, max_seconds=10, clock=clock)
        clock.now = 11
        assert guard.expired()
        assert guard.try_reserve("searches") is None
        assert SECONDS in guard.exhausted_counters()
        assert guard.report(SECONDS).decision == "abort"

    def test_restore_carries_spend_over(self) -> None:
        guard = BudgetGuard({"searches": 5})
        guard.restore({"searches": 4, "unknown": 9})
        assert guard.remaining("searches") == 1

    def test_report_continues_while_capacity_remains(self) -> None:
        assert BudgetGuard({"searches": 5}).report("searches").decision == "continue"


def test_a_zero_cap_is_not_an_exhausted_counter() -> None:
    guard = BudgetGuard({"gap_fill_rounds": 0, "searches": 1})
    assert guard.try_reserve("gap_fill_rounds") is None
    assert guard.exhausted_counters() == []
