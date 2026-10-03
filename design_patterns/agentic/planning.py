"""Planning and decomposition — commit to a plan, then execute it (E06).

Hierarchical Task Network planning, applied to language agents: break the
goal into ordered sub-goals before doing any of them.

The framing that matters (E04): a plan is not one autonomy decision, it
is a *tree* of them. Each step gets its own workflow-or-agent choice, and
a step that is a fixed sequence should be run as one rather than handed
to a model because its siblings needed one.

Where it breaks: when the first step's result invalidates the rest of the
plan. Committing up front is the trade — you get an auditable intention
before any action, and you lose the ability to react to what you find.
"""

from typing import Awaitable, Callable

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

# Executes one planned step, given the step and what came before it.
ExecuteStep = Callable[[str, list[str]], Awaitable[str]]

_PLAN_PROMPT = (
    "Break this task into a short ordered list of concrete steps. "
    "One step per line, no numbering, no commentary.\n\nTask: {task}"
)


class PlanningPattern:
    """Produces an explicit plan, then executes it step by step."""

    name: str = "planning"
    family: PatternFamily = "AG"

    def __init__(self, llm: LLMClient, execute: ExecuteStep, max_steps: int = 6) -> None:
        self._llm = llm
        self._execute = execute
        self._max_steps = max_steps
        self._logger = get_logger("design_patterns")

    async def plan(self, task: str) -> list[str]:
        """Ask for a decomposition and read it back as a list of steps."""
        result = await self._llm.complete(
            CompletionRequest(
                messages=[Message(role="user", content=_PLAN_PROMPT.format(task=task))]
            )
        )
        steps = [line.strip("-•* \t") for line in result.content.splitlines() if line.strip()]
        return steps[: self._max_steps]

    async def run(self, task: str) -> PatternRun:
        """Plan, then execute each step in order."""
        planned = await self.plan(task)
        if not planned:
            return PatternRun(
                pattern=self.name,
                family=self.family,
                succeeded=False,
                steps=[Step(kind="plan", detail="no plan produced")],
            )
        steps = [Step(kind="plan", detail=f"{len(planned)} steps", output="\n".join(planned))]
        completed: list[str] = []
        for index, planned_step in enumerate(planned, start=1):
            outcome = await self._execute(planned_step, list(completed))
            completed.append(outcome)
            steps.append(Step(kind="execute", detail=f"{index}. {planned_step}", output=outcome))
            self._logger.debug("Plan step complete", step=index, of=len(planned))
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=completed[-1] if completed else "",
            steps=steps,
            metadata={"planned_steps": len(planned)},
        )
