"""A loop governor for one agent, built once in its composition root (Factory).

The agent supplies its own caps (from its own config: its tool iterations, its answer budget,
its step cap); everything else — the stagnation
tiers, convergence, the historical baseline over the agent's own turn facts — is the
platform's, from `loop_control.*`.
"""

from collections.abc import Callable
from typing import Any

from agent_platform.config import get_settings
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


__all__ = ["build_loop_governor"]
