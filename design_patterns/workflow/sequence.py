"""Sequence — fixed steps, in order (E04).

The simplest workflow shape and the right default. Every step runs, in
the order written, and the output of one feeds the next.

Reach for it when the steps and their order never vary. Its limitation is
exactly its guarantee: a sequence cannot react. If the right next step
depends on what the last one returned, this is the wrong shape and you
want a branch or an agent.
"""

from typing import Awaitable, Callable

from agentic_commons.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

# One stage: takes what the previous stage produced, returns its own.
Stage = Callable[[str], Awaitable[str]]


class SequencePattern:
    """Runs named stages in order, recording each one."""

    name: str = "sequence"
    family: PatternFamily = "WF"

    def __init__(self, stages: list[tuple[str, Stage]]) -> None:
        self._stages = stages
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Feed the task through every stage in turn."""
        steps: list[Step] = []
        current = task
        for name, stage in self._stages:
            current = await stage(current)
            steps.append(Step(kind="stage", detail=name, output=current))
            self._logger.debug("Sequence stage complete", stage=name)
        return PatternRun(pattern=self.name, family=self.family, output=current, steps=steps)
