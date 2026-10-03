"""Fan-out/fan-in — run independent branches, then join (E04).

Every branch runs; the results are combined. The word doing the work is
*independent*: if one branch needs another's output, they are not
parallel, they are a sequence wearing a costume, and running them
concurrently will produce a race rather than a speed-up.

Concurrency here is real (`asyncio.gather`), so a slow branch costs the
whole join only once rather than once per branch.
"""

import asyncio
from typing import Awaitable, Callable

from agent_platform.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

Branch = Callable[[str], Awaitable[str]]
Join = Callable[[list[str]], str]


class FanOutFanInPattern:
    """Runs branches concurrently and joins their results."""

    name: str = "fan_out_fan_in"
    family: PatternFamily = "WF"

    def __init__(self, branches: list[tuple[str, Branch]], join: Join) -> None:
        self._branches = branches
        self._join = join
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Run every branch, then combine.

        A branch that raises fails the whole join rather than being
        quietly dropped: a partial answer presented as a complete one is
        worse than an error.
        """
        results = await asyncio.gather(*(branch(task) for _, branch in self._branches))
        steps = [
            Step(kind="branch", detail=name, output=output)
            for (name, _), output in zip(self._branches, results)
        ]
        joined = self._join(list(results))
        self._logger.debug("Fan-in complete", branches=len(steps))
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=joined,
            steps=[*steps, Step(kind="join", detail="join", output=joined)],
        )
