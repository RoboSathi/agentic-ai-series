"""Branch — a deterministic predicate picks one fixed path (E04).

A person writes the condition and every possible destination. Which path
runs varies; the set of paths does not.

Reach for it when the condition is a rule someone can read and check. If
deciding requires judgement about unstructured text, a predicate becomes
a pile of string matching that nobody can reason about, and that is the
signal to move to a router with a classifier — or to an agent.
"""

from typing import Awaitable, Callable

from agent_platform.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

Predicate = Callable[[str], bool]
Handler = Callable[[str], Awaitable[str]]


class BranchPattern:
    """First matching predicate wins; an else-branch catches the rest."""

    name: str = "branch"
    family: PatternFamily = "WF"

    def __init__(
        self,
        branches: list[tuple[str, Predicate, Handler]],
        otherwise: tuple[str, Handler] | None = None,
    ) -> None:
        self._branches = branches
        self._otherwise = otherwise
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Evaluate predicates in order and run the first that matches."""
        for label, predicate, handler in self._branches:
            if predicate(task):
                output = await handler(task)
                return self._result(label, output, matched=True)
        if self._otherwise is None:
            return self._result("no branch matched", "", matched=False)
        label, handler = self._otherwise
        return self._result(label, await handler(task), matched=True)

    def _result(self, label: str, output: str, matched: bool) -> PatternRun:
        self._logger.debug("Branch taken", branch=label)
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=output,
            succeeded=matched,
            steps=[Step(kind="branch", detail=label, output=output)],
            metadata={"branch": label},
        )
