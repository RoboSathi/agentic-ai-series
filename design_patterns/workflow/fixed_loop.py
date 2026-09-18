"""Fixed loop — repeat a bounded, known number of times (E04).

The iteration count is decided before the loop starts, so the work is
bounded by construction. That is the whole difference from a ReAct loop,
where the model decides when to stop and something else has to bound it.

Reach for it when you know how many times. If only the model can tell
when the work is done, this is the wrong shape — and E20's watchdog stack
exists because that case needs real controls.
"""

from typing import Awaitable, Callable

from agentic_commons.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

Iteration = Callable[[str, int], Awaitable[str]]


class FixedLoopPattern:
    """Runs an iteration a fixed number of times, optionally stopping early."""

    name: str = "fixed_loop"
    family: PatternFamily = "WF"

    def __init__(
        self,
        iteration: Iteration,
        times: int,
        stop_when: Callable[[str], bool] | None = None,
    ) -> None:
        self._iteration = iteration
        self._times = times
        self._stop_when = stop_when
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Iterate up to the cap, stopping early if the condition is met.

        An early-stop condition does not make this an agent: the cap is
        still fixed, so the worst case is still known.
        """
        steps: list[Step] = []
        current = task
        for index in range(self._times):
            current = await self._iteration(current, index)
            steps.append(Step(kind="iteration", detail=f"pass {index + 1}", output=current))
            if self._stop_when is not None and self._stop_when(current):
                self._logger.debug("Fixed loop stopped early", passes=index + 1)
                break
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=current,
            steps=steps,
            metadata={"passes": len(steps), "cap": self._times},
        )
