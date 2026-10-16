"""A loop governor for one agent, built once in its composition root (Factory).

The agent supplies its own caps (from its own config: its tool iterations, its answer budget,
its step cap); everything else — the stagnation
tiers, convergence, the historical baseline over the agent's own turn facts — is the
platform's, from `loop_control.*`.
"""

from collections.abc import Callable
from typing import Any

from agent_platform.config import get_settings
from agent_platform.loop_control.budget import SECONDS, BudgetGuard
from agent_platform.loop_control.governor import LoopBudget, LoopGovernor
from agent_platform.loop_control.stagnation import FactStepHistory, HistoricalBaseline
from agent_platform.loop_control.types import LoopVerdict


def build_loop_governor(
    budget: LoopBudget,
    *,
    facts: Any = None,
    agent_id: str = "",
    on_exhausted: Callable[[LoopVerdict], None] | None = None,
) -> LoopGovernor:
    """A governor over `budget`; with the agent's facts store, a historical baseline too."""
    settings = get_settings().loop_control.historical
    baseline = (
        HistoricalBaseline(
            FactStepHistory(facts, agent_id), settings.min_samples, settings.sample_size
        )
        if facts is not None and settings.enabled
        else None
    )
    return LoopGovernor(budget, baseline=baseline, on_exhausted=on_exhausted)


def build_shared_loop_governor(
    guard: BudgetGuard,
    step_counter: str,
    *,
    on_exhausted: Callable[[LoopVerdict], None] | None = None,
) -> LoopGovernor:
    """A governor over a budget the loop's workers share, `step_counter` being its step.

    Its caps are the guard's own: the step cap is that counter's, the deadline the guard's
    clock. No historical baseline: a shared budget bounds a run, not a turn, and the agent's
    turn facts are no baseline for it.
    """
    snapshot = guard.snapshot()
    seconds = snapshot[SECONDS].cap if SECONDS in snapshot else None
    budget = LoopBudget(max_steps=int(snapshot[step_counter].cap), max_seconds=seconds)
    return LoopGovernor(budget, on_exhausted=on_exhausted, guard=guard, step_counter=step_counter)


__all__ = ["build_loop_governor", "build_shared_loop_governor"]
