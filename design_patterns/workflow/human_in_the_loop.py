"""Human-in-the-loop — pause at a defined point for a person (E04, E38).

The workflow stops, a person decides, the workflow resumes. The pause is
at a point chosen at design time, which is what separates this from an
agent that escalates when it feels uncertain.

Two things are easy to get wrong.

The pause must be *before* the irreversible step, not after it. A review
that happens once the money has moved is a notification.

And a queue is a staffing decision with a number attached. Little's Law
says L = λW: at 100 escalations an hour needing two minutes each, that is
3.33 reviewer-hours per hour — more than three full-time reviewers just
to break even. Escalating more is not free, and a queue that arrives
faster than it drains grows without bound rather than getting slower.
"""

from typing import Awaitable, Callable

from agent_platform.logging import get_logger
from agent_platform.observability import LittlesLawQueueModel
from design_patterns.types import PatternFamily, PatternRun, Step

Prepare = Callable[[str], Awaitable[str]]
Decide = Callable[[str], Awaitable[bool]]
Apply = Callable[[str], Awaitable[str]]


class HumanInTheLoopPattern:
    """Prepare, pause for a decision, then apply or abandon."""

    name: str = "human_in_the_loop"
    family: PatternFamily = "WF"

    def __init__(
        self,
        prepare: Prepare,
        decide: Decide,
        apply: Apply,
        on_reject: Apply | None = None,
    ) -> None:
        self._prepare = prepare
        self._decide = decide
        self._apply = apply
        self._on_reject = on_reject
        self._queue_model = LittlesLawQueueModel()
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Prepare a proposal, put it to a human, act on the answer."""
        proposal = await self._prepare(task)
        steps = [Step(kind="prepare", detail="proposal", output=proposal)]

        approved = await self._decide(proposal)
        steps.append(Step(kind="review", detail="approved" if approved else "rejected"))
        self._logger.info("Human decision recorded", approved=approved)

        if not approved:
            rejected = await self._on_reject(proposal) if self._on_reject else ""
            steps.append(Step(kind="abandon", detail="not applied", output=rejected))
            return PatternRun(
                pattern=self.name,
                family=self.family,
                output=rejected,
                steps=steps,
                succeeded=False,
                metadata={"approved": False},
            )

        applied = await self._apply(proposal)
        steps.append(Step(kind="apply", detail="applied", output=applied))
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=applied,
            steps=steps,
            metadata={"approved": True},
        )

    def reviewers_required(self, arrivals_per_hour: float, minutes_each: float) -> float:
        """Reviewer-hours per hour this escalation rate implies (E38).

        Exposed on the pattern because the number belongs next to the
        decision to escalate, not in a capacity spreadsheet nobody reads.
        """
        return self._queue_model.required_capacity(arrivals_per_hour, minutes_each)
