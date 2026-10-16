"""The loop governor: one authority, one verdict, one checkpointable state.

The clock is a counter the test moves, similarity is word overlap, and the historical
baseline reads a fixed list — so every verdict here is a fixed expectation. Under test: which
mechanism decides, in what order, what it says, and that a restored governor carries on
exactly where it stopped.
"""

import asyncio

import pytest

from agent_platform.context.similarity import LexicalSimilarity
from agent_platform.loop_control import (
    BudgetGuard,
    IllegalTransitionError,
    LoopBudget,
    LoopGovernor,
    LoopState,
    LoopStateMachine,
    build_shared_loop_governor,
    canonical_fingerprint,
    route_on_verdict,
)
from agent_platform.loop_control.stagnation import CycleDetector, HistoricalBaseline
from agent_platform.tools.types import ToolCall


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _History:
    def __init__(self, steps: list[int]) -> None:
        self._steps = steps

    def recent_steps(self, limit: int) -> list[int]:
        return self._steps[:limit]


def _governor(budget: LoopBudget | None = None, **kwargs: object) -> LoopGovernor:
    governor = LoopGovernor(
        budget or LoopBudget(max_steps=3), similarity=LexicalSimilarity(), **kwargs
    )  # type: ignore[arg-type]
    governor.begin_turn()
    return governor


def _call(**params: object) -> ToolCall:
    return ToolCall(action="orders.lookup", params=dict(params))


class TestTheStepCapAndBudgets:
    def test_steps_within_the_cap_continue(self) -> None:
        governor = _governor()
        assert [governor.begin_step().decision for _ in range(3)] == ["continue"] * 3

    def test_the_step_past_the_cap_finalizes(self) -> None:
        governor = _governor()
        for _ in range(3):
            governor.begin_step()
        verdict = governor.begin_step()
        assert verdict.decision == "finalize" and verdict.layer == "step_count"
        assert verdict.reason_code == "MAX_TOOL_ITERATIONS"
        assert verdict.budgets["steps"].cap == 3

    def test_a_spent_token_budget_finalizes(self) -> None:
        governor = _governor(LoopBudget(max_steps=5, max_tokens=100))
        governor.begin_step()
        governor.after_model(tokens=150)
        verdict = governor.begin_step()
        assert verdict.layer == "token_budget" and verdict.stops

    def test_the_next_calls_output_is_capped_by_the_tokens_left(self) -> None:
        governor = _governor(LoopBudget(max_steps=5, max_tokens=100))
        governor.after_model(tokens=70)
        assert governor.call_limits(configured_max_tokens=1000).max_tokens == 30

    def test_the_deadline_comes_before_every_other_budget(self) -> None:
        clock = _Clock()
        governor = _governor(LoopBudget(max_steps=5, max_seconds=10, max_tokens=1), clock=clock)
        governor.after_model(tokens=5)
        clock.now = 11
        verdict = governor.begin_step()
        assert verdict.layer == "wall_clock" and verdict.reason_code == "DEADLINE_REACHED"

    async def test_a_call_still_running_at_the_deadline_is_cancelled(self) -> None:
        governor = _governor(LoopBudget(max_steps=5, call_timeout_seconds=0.05))
        cancelled = asyncio.Event()

        async def slow() -> None:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(TimeoutError):
            async with governor.deadline():
                await slow()
        assert cancelled.is_set()


class TestRepeats:
    async def test_an_exact_repeat_is_suppressed_with_the_earlier_result(self) -> None:
        governor = _governor()
        await governor.before_action(_call(order_id=1))
        await governor.after_action(_call(order_id=1), "delivered", True)
        verdict = await governor.before_action(_call(order_id=1))
        assert verdict.decision == "suppress" and verdict.layer == "duplicate_action"
        assert "delivered" in governor.suppression_message(verdict)

    def test_canonical_form_ignores_case_order_whitespace_and_number_format(self) -> None:
        a = ToolCall(action="Orders.Lookup", params={"id": "10001 ", "q": "Late  Parcel"})
        b = ToolCall(action="orders.lookup", params={"q": "late parcel", "id": 10001.0})
        assert canonical_fingerprint(a) == canonical_fingerprint(b)

    async def test_a_failure_worth_retrying_is_forgotten(self) -> None:
        governor = _governor()
        await governor.before_action(_call(order_id=1))
        governor.forget(_call(order_id=1))
        assert (await governor.before_action(_call(order_id=1))).decision == "continue"

    async def test_the_same_failed_step_in_other_words_is_suppressed(self) -> None:
        governor = _governor()
        failed = ToolCall(action="web.search", params={"query": "pune weather tomorrow forecast"})
        await governor.before_action(failed)
        await governor.after_action(failed, "search provider down", False)
        again = ToolCall(action="web.search", params={"query": "forecast pune weather tomorrow"})
        verdict = await governor.before_action(again)
        assert verdict.layer == "semantic_stagnation" and verdict.decision == "suppress"

    def test_a_repeating_block_is_a_cycle(self) -> None:
        detector = CycleDetector(max_period=3, repeats=2)
        assert detector.period(["a", "b", "a", "b"]) == 2
        assert detector.period(["a", "b", "c", "a"]) is None

    async def test_a_model_stuck_proposing_a_suppressed_call_is_stopped(self) -> None:
        governor = _governor(LoopBudget(max_steps=10))
        for _ in range(3):
            governor.begin_step()
            await governor.before_action(_call(order_id=1))
        verdict = governor.begin_step()
        assert verdict.decision == "finalize" and verdict.layer == "cycle"


class TestConvergence:
    async def test_steps_that_add_nothing_new_plateau(self) -> None:
        governor = _governor(LoopBudget(max_steps=10))
        for index in range(4):
            call = _call(order_id=index)
            governor.begin_step()
            await governor.before_action(call)
            await governor.after_action(call, "order delivered on monday", True)
        verdict = governor.begin_step()
        assert verdict.layer == "convergence" and verdict.reason_code == "NO_PROGRESS"


class TestASharedBudget:
    """A governor over a budget its loop's workers share: one of its counters is the step."""

    def _shared(self, clock: _Clock | None = None) -> tuple[LoopGovernor, BudgetGuard]:
        guard = BudgetGuard({"rounds": 2, "searches": 5}, max_seconds=10, clock=clock or _Clock())
        governor = build_shared_loop_governor(guard, "rounds")
        governor.begin_turn()
        return governor, guard

    def test_a_step_is_charged_to_the_shared_step_counter(self) -> None:
        governor, guard = self._shared()
        governor.begin_step()
        assert guard.snapshot()["rounds"].spent == 1 and governor.max_steps == 2

    def test_a_grant_on_the_step_counter_raises_the_cap(self) -> None:
        governor, guard = self._shared()
        guard.extend("rounds", 1)
        assert governor.max_steps == 3

    def test_the_step_past_the_shared_cap_finalizes(self) -> None:
        governor, _ = self._shared()
        for _ in range(2):
            assert governor.begin_step().decision == "continue"
        verdict = governor.begin_step()
        assert verdict.layer == "step_count" and verdict.budgets["rounds"].cap == 2

    def test_the_shared_clock_is_the_deadline(self) -> None:
        clock = _Clock()
        governor, _ = self._shared(clock)
        clock.now = 11
        assert governor.check().layer == "wall_clock"

    def test_check_does_not_take_a_step(self) -> None:
        governor, guard = self._shared()
        governor.check()
        assert guard.snapshot()["rounds"].spent == 0

    async def test_rounds_that_add_nothing_new_plateau(self) -> None:
        governor, guard = self._shared()
        guard.extend("rounds", 5)
        assert await governor.record_progress(["the order shipped on monday"]) == 1.0
        for _ in range(3):
            governor.begin_step()
            await governor.record_progress([])
        verdict = governor.check()
        assert verdict.decision == "finalize" and verdict.layer == "convergence"

    async def test_a_round_with_something_new_is_progress(self) -> None:
        governor, _ = self._shared()
        await governor.record_progress(["the order shipped on monday"])
        gain = await governor.record_progress(["a refund was issued to the card"])
        assert gain > 0.5

    def test_restoring_leaves_the_shared_budget_to_its_owner(self) -> None:
        governor, guard = self._shared()
        governor.begin_step()
        saved = governor.state
        guard.charge("searches", 3)
        governor.restore(saved)
        assert guard.snapshot()["searches"].spent == 3 and governor.state.step == 1


class TestTheHistoricalBaseline:
    def test_an_unusually_long_turn_is_flagged_not_stopped(self) -> None:
        baseline = HistoricalBaseline(_History([2, 3] * 15), min_samples=20, sample_size=200)
        governor = _governor(LoopBudget(max_steps=20), baseline=baseline)
        for _ in range(9):
            verdict = governor.begin_step()
        assert verdict.decision == "continue"
        assert any(check.reason_code == "UNUSUAL_LENGTH" for check in verdict.checks)

    def test_too_little_history_says_nothing(self) -> None:
        assert HistoricalBaseline(_History([3]), 20, 200).z_score(30) is None


class TestStateSurvivesACheckpoint:
    async def test_a_repeat_is_still_a_repeat_after_a_restore(self) -> None:
        governor = _governor()
        governor.begin_step()
        await governor.before_action(_call(order_id=1))
        await governor.after_action(_call(order_id=1), "delivered", True)
        saved = LoopState.model_validate_json(governor.state.model_dump_json())
        resumed = LoopGovernor(LoopBudget(max_steps=3), similarity=LexicalSimilarity())
        resumed.restore(saved)
        assert (await resumed.before_action(_call(order_id=1))).decision == "suppress"

    def test_the_seconds_already_used_are_not_given_back(self) -> None:
        clock = _Clock()
        governor = _governor(LoopBudget(max_steps=5, max_seconds=10), clock=clock)
        clock.now = 8
        saved = governor.state
        later = _Clock()
        resumed = LoopGovernor(
            LoopBudget(max_steps=5, max_seconds=10), similarity=LexicalSimilarity(), clock=later
        )
        resumed.restore(saved)
        assert resumed.time_left() == pytest.approx(2)

    def test_the_turn_ends_in_one_terminal_outcome(self) -> None:
        governor = _governor()
        governor.begin_step()
        state = governor.finish("answered")
        assert state.outcome == "answered" and state.phase == "DONE"

    def test_an_illegal_phase_change_is_refused(self) -> None:
        with pytest.raises(IllegalTransitionError):
            LoopStateMachine().transition(LoopState(phase="IDLE"), "WAITING_TOOL")


class TestRouting:
    def test_routers_read_only_the_verdict(self) -> None:
        governor = _governor(LoopBudget(max_steps=1))
        governor.begin_step()
        stopped = governor.begin_step()
        routes = {"finalize": "answer"}
        assert route_on_verdict(stopped, routes, "next") == "answer"
        assert route_on_verdict(None, routes, "next") == "next"
