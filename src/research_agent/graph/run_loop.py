"""LOOP CONTROL for a research run: every round's decision, as the platform card.

The run graph decides in its routing edges (`after_reflect`, `after_audit`, `fan_out`),
which run off the event loop and must not draw cards. So the node before each edge works
out the same decision and reports it here: the criteria it weighed (every budget meter,
the round, the gaps), the step it chose, and why. Words come from
`research_agent.run_loop`; a `because` that is not a key there is shown as written (a stop
sentence the reflector already composed).
"""

from typing import Any

from agent_platform.loop_control import BudgetGuard
from agent_platform.observability import StageReporter
from agent_platform.observability.step_events import StepEvents
from research_agent.config.settings import RunLoopWords
from research_agent.domain.budget import ROUNDS


class RunLoopReport:
    """Reports one research run's loop decisions on its live feed."""

    def __init__(self, reporter: StageReporter | None, words: RunLoopWords) -> None:
        self._reporter = reporter
        self._words = words

    def report(
        self,
        budget: BudgetGuard,
        *,
        round_number: int,
        next_step: str,
        because: str,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One LOOP CONTROL card; the fields it showed, for the log."""
        steps = StepEvents(self._reporter, max_steps=int(budget.snapshot()[ROUNDS].cap))
        return steps.loop_control(
            round_number=round_number,
            criteria={**(criteria or {}), **_meters(budget)},
            next_step=self._words.next.get(next_step, next_step),
            because=because,
            because_words=self._words.because,
        )


def _meters(budget: BudgetGuard) -> dict[str, str]:
    """Every budget meter, spent of cap — what decides whether another round may run."""
    return {name: f"{c.spent:g} of {c.cap:g}" for name, c in budget.snapshot().items()}


__all__ = ["RunLoopReport"]
