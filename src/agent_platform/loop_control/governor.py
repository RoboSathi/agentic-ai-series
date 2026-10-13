"""The loop governor: the one authority over a tool loop's lifecycle (Facade).

An agent's loop used to be governed by pieces it assembled itself — a step cap here, a
duplicate check there, a budget guard in one agent and not the next, a router and a card
reporter each restating the same rule. `LoopGovernor` is all of it behind one interface:

    begin_turn(goal)       a new turn: phase PLANNING, the clock and counters as restored
    begin_step()           → verdict: may the loop take another step at all?
    call_limits()          → the next call's max_tokens and timeout, from what is left
    deadline(seconds)      an async scope that *cancels* a call still running when time is up
    after_model(result)    its tokens and cost charged
    before_action(call)    → verdict: run it, or suppress it (a repeat, by any tier)
    after_action(...)      what it returned, remembered; the evidence it added, measured
    finish(outcome)        the turn's one terminal outcome

Every answer is a `LoopVerdict`; the node writes it into graph state and every router reads
`route_on_verdict` — so a rule is written once. The governor logs each intervention once;
the node draws the LOOP CONTROL card from the verdict (`StepEvents.loop_verdict`), since a
router runs off the event loop and must not draw.

State is a `LoopState`, kept by the agent in its checkpointed graph state; `restore` puts a
resumed governor back exactly where it was, seconds included.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from agent_platform.config import get_settings
from agent_platform.context.similarity import TextSimilarity, shared_similarity
from agent_platform.logging import get_logger
from agent_platform.loop_control.budget import SECONDS, BudgetGuard
from agent_platform.loop_control.convergence import ConvergenceMonitor
from agent_platform.loop_control.stagnation import (
    CycleDetector,
    HistoricalBaseline,
    SemanticTier,
    action_fingerprint,
    canonical_fingerprint,
    describe,
)
from agent_platform.loop_control.state import ActionRecord, LoopState, LoopStateMachine
from agent_platform.loop_control.types import (
    BudgetLine,
    ControlCheck,
    LoopDecision,
    LoopPhase,
    LoopVerdict,
    TerminalOutcome,
    WatchdogLayer,
)
from agent_platform.observability.units import report_loop
from agent_platform.tools.types import ToolCall

# The counters every governed loop has (a cap of 0 means "not budgeted").
STEPS, TOOL_CALLS, TOKENS, COST = "steps", "tool_calls", "tokens", "cost_usd"
_CONTINUE_DECISION = "continue"
_COUNTER_LAYERS: dict[str, WatchdogLayer] = {
    STEPS: "step_count",
    TOOL_CALLS: "tool_budget",
    TOKENS: "token_budget",
    COST: "cost_budget",
    SECONDS: "wall_clock",
}
_CONTINUE = "LOOP_CONTROL_CONTINUE"


@dataclass(frozen=True)
class LoopBudget:
    """One agent's caps, from its own config; None leaves a counter unbudgeted."""

    max_steps: int
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    call_timeout_seconds: float | None = None


@dataclass(frozen=True)
class CallLimits:
    """What the next model call may use: its output cap and how long it may run."""

    max_tokens: int | None
    timeout_seconds: float


class LoopGovernor:
    """Budgets, deadline, stagnation, convergence and the historical baseline, as one authority."""

    def __init__(
        self,
        budget: LoopBudget,
        *,
        similarity: TextSimilarity | None = None,
        baseline: HistoricalBaseline | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_exhausted: Callable[[LoopVerdict], None] | None = None,
    ) -> None:
        settings = get_settings().loop_control
        self._settings = settings
        self._budget = budget
        self._clock = clock
        self._on_exhausted = on_exhausted
        self._similarity = similarity or shared_similarity()
        self._semantic = SemanticTier(self._similarity, settings.stagnation.semantic_threshold)
        self._cycles = CycleDetector(
            settings.stagnation.cycle_max_period, settings.stagnation.cycle_repeats
        )
        self._convergence = ConvergenceMonitor(
            self._similarity, settings.convergence, settings.stagnation.history_window
        )
        self._baseline = baseline if settings.historical.enabled else None
        self._machine = LoopStateMachine()
        self._logger = get_logger("loop_control")
        self._state = LoopState()
        self._guard = self._new_guard()

    # -- lifecycle --------------------------------------------------------------------

    @property
    def state(self) -> LoopState:
        """The state to keep in the graph (a copy: the governor keeps its own)."""
        self._sync()
        return self._state.model_copy(deep=True)

    @property
    def verdict(self) -> LoopVerdict | None:
        return self._state.verdict

    @property
    def max_steps(self) -> int:
        return self._budget.max_steps

    def restore(self, state: LoopState) -> None:
        """Continue a loop from its checkpointed state: counters, seconds, actions, phase."""
        self._state = state.model_copy(deep=True)
        self._guard = self._new_guard()
        self._guard.restore(self._state.spent, elapsed=self._state.seconds)

    def begin_turn(self) -> None:
        """A fresh turn: clean counters and memory, phase PLANNING, the cap announced."""
        self._state = LoopState()
        self._guard = self._new_guard()
        self._move("PLANNING")
        report_loop(self._budget.max_steps, False)

    def begin_step(self) -> LoopVerdict:
        """Charge a step and say whether the loop may take it."""
        self._state.step += 1
        self._guard.charge(STEPS, 1)
        self._move("PLANNING")
        verdict = self._step_verdict()
        report_loop(self._budget.max_steps, verdict.layer == "step_count")
        return self._decided(verdict)

    def call_limits(self, configured_max_tokens: int | None = None) -> CallLimits:
        """The next call's output cap (never past the token budget) and timeout (never past
        the turn's deadline)."""
        cap = configured_max_tokens
        if self._budget.max_tokens:
            left = int(self._guard.remaining(TOKENS))
            cap = min(cap, left) if cap else left
        return CallLimits(max_tokens=cap, timeout_seconds=self._call_seconds())

    def spent(self, counter: str) -> float:
        """What `counter` has spent this turn (`steps` counts the steps taken)."""
        if counter == STEPS:
            return float(self._state.step)
        snapshot = self._guard.snapshot()
        return snapshot[counter].spent if counter in snapshot else 0.0

    def remaining(self, counter: str) -> float:
        """What `counter` has left (unbounded counters: infinite)."""
        if counter == STEPS:
            return float(max(self._budget.max_steps - self._state.step, 0))
        snapshot = self._guard.snapshot()
        return self._guard.remaining(counter) if counter in snapshot else float("inf")

    def elapsed(self) -> float:
        """Seconds this turn (or run, across resumes) has used."""
        return self._elapsed()

    def exhausted_counters(self) -> list[str]:
        """Every counter that has run out, the clock included."""
        names = [c for c in (STEPS, TOOL_CALLS, TOKENS, COST, SECONDS) if self._exhausted(c)]
        return names

    def time_left(self) -> float:
        """Seconds the turn has left (the per-call limit when it has no deadline)."""
        if self._budget.max_seconds is None:
            return self._call_seconds()
        return max(self._budget.max_seconds - self._elapsed(), 0.0)

    def can_call(self) -> bool:
        """Whether enough time is left to start a model call at all."""
        return self.time_left() >= self._settings.min_call_seconds

    @asynccontextmanager
    async def deadline(self, seconds: float | None = None) -> AsyncIterator[None]:
        """Run a call under the turn's deadline: still running when it passes, it is cancelled
        (`TimeoutError` out of the block), not merely ignored."""
        limit = min(seconds or self._call_seconds(), self._call_seconds())
        async with asyncio.timeout(limit):
            yield

    def after_model(self, tokens: int, cost_usd: float = 0.0) -> None:
        """Charge a model call's tokens and cost."""
        self._move("ACTING")
        if self._budget.max_tokens:
            self._guard.charge(TOKENS, tokens)
        if self._budget.max_cost_usd:
            self._guard.charge(COST, cost_usd)

    async def before_action(self, call: ToolCall) -> LoopVerdict:
        """Run this action, or suppress it as a repeat (exact, canonical or semantic)."""
        exact, canonical = action_fingerprint(call), canonical_fingerprint(call)
        verdict = await self._action_verdict(call, exact, canonical)
        record = ActionRecord(action=call.action, fingerprint=exact, canonical=canonical)
        if verdict.decision == "suppress":
            record = record.model_copy(
                update={"suppressed": True, "outcome": verdict.previous_result}
            )
            self._state.actions.append(record)
        elif verdict.decision == _CONTINUE_DECISION:
            self._move("WAITING_TOOL")
            if self._budget.max_tool_calls:
                self._guard.charge(TOOL_CALLS, 1)
            self._state.actions.append(record)
        # Anything else (a spent tool budget) stopped the call: it neither ran nor is charged,
        # so the counter never reads past its cap.
        return self._decided(verdict, quiet=verdict.decision == "continue")

    async def after_action(self, call: ToolCall, outcome: str, succeeded: bool) -> None:
        """What the action returned (or why it failed), and the new evidence it added."""
        fingerprint = action_fingerprint(call)
        for record in reversed(self._state.actions):
            if record.fingerprint == fingerprint and not record.suppressed:
                record.outcome, record.succeeded = outcome, succeeded
                break
        gain = await self._convergence.gain(outcome, self._state.evidence) if succeeded else 0.0
        self._state.gains.append(gain)
        if succeeded:
            self._state.evidence = [*self._state.evidence, outcome][-self._window :]
        self._move("ACTING")

    def forget(self, call: ToolCall) -> None:
        """An action whose failure is worth one more try (a write): not remembered at all."""
        fingerprint = action_fingerprint(call)
        for index in range(len(self._state.actions) - 1, -1, -1):
            record = self._state.actions[index]
            if record.fingerprint == fingerprint and not record.suppressed:
                del self._state.actions[index]
                return

    def record_refusal(self, call: ToolCall, reason: str) -> None:
        """An action the Gatekeeper refused: as repeatable, and as pointless to repeat."""
        for record in reversed(self._state.actions):
            if record.fingerprint == action_fingerprint(call) and not record.suppressed:
                record.outcome, record.succeeded = reason, False
                return

    def suppression_message(self, verdict: LoopVerdict) -> str:
        """What the model is told instead of a result, for a suppressed action."""
        words = self._settings.messages
        previous = verdict.previous_result
        if verdict.reason_code == "SEMANTIC_STAGNATION":
            return words.suppressed_similar.format(previous=previous)
        record = next(
            (
                r
                for r in self._state.actions
                if r.fingerprint == verdict.fingerprint and not r.suppressed
            ),
            None,
        )
        template = (
            words.suppressed_success
            if record is None or record.succeeded
            else words.suppressed_refused
        )
        return template.format(previous=previous)

    def finish(self, outcome: TerminalOutcome) -> LoopState:
        """The turn's one terminal outcome; the phase moves to DONE (or FAILED)."""
        self._state.outcome = outcome
        self._move("FAILED" if outcome == "failed" else "DONE")
        self._logger.info(
            "Loop finished",
            outcome=outcome,
            steps=self._state.step,
            budgets={k: v.model_dump() for k, v in self._budgets().items()},
        )
        return self.state

    # -- verdicts -----------------------------------------------------------------------

    def _step_verdict(self) -> LoopVerdict:
        """Deadline, then each budget, then a cycle, then a plateau; else continue."""
        for counter in (SECONDS, STEPS, TOKENS, COST):
            if self._exhausted(counter):
                return self._budget_verdict(counter)
        attempted = [record.canonical for record in self._state.actions]
        period = self._cycles.period(attempted)
        if period is not None:
            return self._verdict("finalize", "cycle", "ACTION_CYCLE", period=period)
        if self._convergence.plateaued(self._state.gains):
            return self._verdict(
                "finalize", "convergence", "NO_PROGRESS", window=self._settings.convergence.window
            )
        return self._with_baseline(self._verdict("continue", None, _CONTINUE))

    async def _action_verdict(self, call: ToolCall, exact: str, canonical: str) -> LoopVerdict:
        if self._exhausted(TOOL_CALLS):
            return self._budget_verdict(TOOL_CALLS, decision="finalize")
        done = [r for r in self._state.actions if not r.suppressed]
        same = next((r for r in done if r.fingerprint == exact), None)
        if same is not None:
            return self._repeat("duplicate_action", "DUPLICATE_ACTION", exact, same)
        alike = next((r for r in done if r.canonical == canonical), None)
        if alike is not None:
            return self._repeat("duplicate_action", "CANONICAL_DUPLICATE", exact, alike)
        return await self._semantic_verdict(call, exact, done)

    async def _semantic_verdict(
        self, call: ToolCall, exact: str, done: list[ActionRecord]
    ) -> LoopVerdict:
        failed = [r for r in done if r.outcome and not r.succeeded]
        if not self._settings.stagnation.semantic or not failed:
            return self._verdict("continue", None, _CONTINUE, fingerprint=exact)
        texts = [f"{r.action} {r.canonical}" for r in failed]
        score, index = await self._semantic.closest(describe(call), texts)
        if self._semantic.repeats(score):
            return self._repeat("semantic_stagnation", "SEMANTIC_STAGNATION", exact, failed[index])
        return self._verdict("continue", None, _CONTINUE, fingerprint=exact)

    def _repeat(
        self, layer: WatchdogLayer, code: str, exact: str, previous: ActionRecord
    ) -> LoopVerdict:
        return self._verdict(
            "suppress", layer, code, fingerprint=previous.fingerprint, previous=previous.outcome
        ).model_copy(update={"fingerprint": exact})

    def _budget_verdict(self, counter: str, decision: LoopDecision = "finalize") -> LoopVerdict:
        line = self._budgets()[counter]
        code = "DEADLINE_REACHED" if counter == SECONDS else "BUDGET_EXHAUSTED"
        if counter == STEPS:
            code = "MAX_TOOL_ITERATIONS"
        return self._verdict(
            decision,
            _COUNTER_LAYERS[counter],
            code,
            counter=counter,
            spent=line.spent,
            cap=line.cap,
        )

    def _with_baseline(self, verdict: LoopVerdict) -> LoopVerdict:
        """The historical baseline's flag, as a failed check on a verdict it does not change."""
        if self._baseline is None:
            return verdict
        z = self._baseline.z_score(self._state.step)
        if z is None or z < self._settings.historical.z_threshold:
            return verdict
        check = self._check("historical", "UNUSUAL_LENGTH", "FAIL", z=z)
        return verdict.model_copy(update={"checks": [*verdict.checks, check]})

    def _verdict(
        self,
        decision: LoopDecision,
        layer: WatchdogLayer | None,
        code: str,
        *,
        fingerprint: str = "",
        previous: str = "",
        **values: Any,
    ) -> LoopVerdict:
        status = "PASS" if decision == "continue" else "FAIL"
        checks = [self._check(layer or "step_count", code, status, **values)]
        return LoopVerdict(
            decision=decision,
            layer=layer,
            reason_code=code,
            reason=checks[0].reason,
            checks=checks,
            round_number=self._state.step,
            max_rounds=self._budget.max_steps,
            fingerprint=fingerprint,
            previous_result=previous,
            budgets=self._budgets(),
        )

    def _check(self, layer: str, code: str, status: str, **values: Any) -> ControlCheck:
        template = self._settings.reasons[code]
        values = {"step": self._state.step, "max": self._budget.max_steps, **values}
        return ControlCheck(
            check_id=f"loop_control.{layer}",
            name=layer,
            status=status,  # type: ignore[arg-type]
            reason_code=code,
            reason=template.format(**values),
            observed=values,
        )

    def _decided(self, verdict: LoopVerdict, *, quiet: bool = False) -> LoopVerdict:
        """Keep the verdict; log it once if it intervened; tell the owner a budget ran out."""
        self._state.verdict = verdict
        self._sync()
        if verdict.intervened:
            self._logger.info(
                "Loop verdict",
                decision=verdict.decision,
                layer=verdict.layer,
                reason_code=verdict.reason_code,
                reason=verdict.reason,
                step=verdict.round_number,
                fingerprint=verdict.fingerprint,
                budgets={k: v.model_dump() for k, v in verdict.budgets.items()},
            )
            if verdict.layer in _COUNTER_LAYERS.values() and self._on_exhausted is not None:
                self._on_exhausted(verdict)
        elif not quiet:
            self._logger.debug("Loop verdict", decision="continue", step=verdict.round_number)
        return verdict

    # -- counters -----------------------------------------------------------------------

    def _new_guard(self) -> BudgetGuard:
        caps: dict[str, float] = {STEPS: self._budget.max_steps}
        for counter, cap in (
            (TOOL_CALLS, self._budget.max_tool_calls),
            (TOKENS, self._budget.max_tokens),
            (COST, self._budget.max_cost_usd),
        ):
            if cap:
                caps[counter] = cap
        return BudgetGuard(
            caps, max_seconds=self._budget.max_seconds, clock=self._clock, layers=_COUNTER_LAYERS
        )

    def _exhausted(self, counter: str) -> bool:
        if counter == SECONDS:
            return self._budget.max_seconds is not None and self._guard.expired()
        if counter == STEPS:
            return self._state.step > self._budget.max_steps
        snapshot = self._guard.snapshot()
        return counter in snapshot and snapshot[counter].exhausted

    def _budgets(self) -> dict[str, BudgetLine]:
        return {
            name: BudgetLine(spent=round(state.spent, 6), cap=state.cap)
            for name, state in self._guard.snapshot().items()
        }

    def _elapsed(self) -> float:
        return self._guard.elapsed()

    def _call_seconds(self) -> float:
        per_call = self._budget.call_timeout_seconds or self._settings.call_timeout_seconds
        if self._budget.max_seconds is None:
            return per_call
        return max(min(per_call, self._budget.max_seconds - self._elapsed()), 0.0)

    def _sync(self) -> None:
        self._state.spent = {
            name: state.spent for name, state in self._guard.snapshot().items() if name != SECONDS
        }
        self._state.seconds = round(self._elapsed(), 3)

    @property
    def _window(self) -> int:
        return self._settings.stagnation.history_window

    def _move(self, target: LoopPhase) -> None:
        self._machine.transition(self._state, target)


def route_on_verdict(
    verdict: LoopVerdict | None, routes: Mapping[LoopDecision, str], default: str
) -> str:
    """The one routing rule every agent's graph uses: the verdict's decision → the next node."""
    if verdict is None:
        return default
    return routes.get(verdict.decision, default)


__all__ = [
    "COST",
    "STEPS",
    "TOKENS",
    "TOOL_CALLS",
    "CallLimits",
    "LoopBudget",
    "LoopGovernor",
    "route_on_verdict",
]
