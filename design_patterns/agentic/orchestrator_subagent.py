"""Orchestrator-subagent — one agent plans, others execute (E07).

Divide-and-conquer: the orchestrator owns the plan and delegates bounded
sub-tasks; each subagent owns its sub-task and has no visibility into the
parent's full context.

That isolation is the feature. A subagent working on one sub-task cannot
be distracted by the rest of the problem, and its context stays small.
It is also the cost: the orchestrator must pass down everything the
subagent needs, and anything it forgets is simply absent.

Taught here at pattern-shape level. The economics — when the coordination
cost exceeds the parallelism benefit — are E40/E43's subject, and the
topology choices are E41's. This is deliberately the shape only.
"""

import asyncio
from typing import Awaitable, Callable

from agent_platform.logging import get_logger
from agent_platform.resilience import ResourcePoolBulkhead
from design_patterns.types import PatternFamily, PatternRun, Step

# A subagent: given one bounded sub-task, returns its result. It is given
# a string, not the parent's state, because that is the isolation.
Subagent = Callable[[str], Awaitable[str]]
Decompose = Callable[[str], Awaitable[list[str]]]
Synthesize = Callable[[str, list[str]], Awaitable[str]]


class OrchestratorSubagentPattern:
    """Decomposes a task, runs subagents concurrently, synthesises the result."""

    name: str = "orchestrator_subagent"
    family: PatternFamily = "AG"

    def __init__(
        self,
        decompose: Decompose,
        subagent: Subagent,
        synthesize: Synthesize,
        bulkhead: ResourcePoolBulkhead | None = None,
    ) -> None:
        self._decompose = decompose
        self._subagent = subagent
        self._synthesize = synthesize
        self._bulkhead = bulkhead or ResourcePoolBulkhead()
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Split the work, run it in parallel, then combine."""
        subtasks = await self._decompose(task)
        if not subtasks:
            return PatternRun(
                pattern=self.name,
                family=self.family,
                succeeded=False,
                steps=[Step(kind="decompose", detail="nothing to delegate")],
            )
        steps = [Step(kind="decompose", detail=f"{len(subtasks)} sub-tasks")]
        results = await asyncio.gather(*(self._delegate(sub) for sub in subtasks))
        steps += [
            Step(kind="subagent", detail=sub, output=output)
            for sub, output in zip(subtasks, results)
        ]
        synthesis = await self._synthesize(task, list(results))
        steps.append(Step(kind="synthesize", detail="combined", output=synthesis))
        self._logger.debug("Orchestration complete", subagents=len(subtasks))
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=synthesis,
            steps=steps,
            metadata={"subagents": len(subtasks)},
        )

    async def _delegate(self, subtask: str) -> str:
        """Run one subagent, bounded by the pool.

        Unbounded fan-out is how a decomposition into two hundred
        sub-tasks takes down the model endpoint it depends on.
        """
        async with self._bulkhead.acquire("subagents"):
            return await self._subagent(subtask)
