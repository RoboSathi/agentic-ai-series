"""ReAct — reason, act, observe, repeat (E05).

Yao et al. (2022): interleaving explicit reasoning with actions beats
either reasoning alone (chain-of-thought, which cannot look anything up)
or acting alone (which cannot plan). The trajectory — the accumulated
thought/action/observation triples — is the unit of work, and it is also
the audit record.

This is the atomic pattern the rest of the module composes. Planning
sequences several of these; reflection critiques a finished one;
orchestration runs several in parallel.

Where it fails, and why the rest of the series exists: the trajectory
grows past a useful context (E17), and there is no natural stopping point
(E20). The step cap here is the crudest possible answer to the second
problem, and the fact that it is crude is the point — a bound you can
state is worth more than a heuristic you cannot.
"""

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import get_logger
from agent_platform.tools import NamespacedToolRegistry
from design_patterns.agentic._invoke import invoke_tool
from design_patterns.types import PatternFamily, PatternRun, Step


class ReActPattern:
    """Loops reason→act→observe until the model answers or the cap is hit."""

    name: str = "react"
    family: PatternFamily = "AG"

    def __init__(
        self, llm: LLMClient, registry: NamespacedToolRegistry, max_steps: int = 5
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Run the loop, recording the whole trajectory."""
        messages = [Message(role="user", content=task)]
        steps: list[Step] = []
        tools = self._registry.to_openai_tools()

        for index in range(self._max_steps):
            result = await self._llm.complete(CompletionRequest(messages=messages, tools=tools))
            steps.append(
                Step(
                    kind="thought",
                    detail=f"step {index + 1}",
                    output=result.content,
                    tokens=result.usage.total_tokens,
                )
            )
            if not result.tool_calls:
                return PatternRun(
                    pattern=self.name,
                    family=self.family,
                    output=result.content,
                    steps=steps,
                    metadata={"steps_taken": index + 1, "hit_cap": False},
                )
            messages.append(
                Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
            )
            for requested in result.tool_calls:
                observation = await self._observe(requested.name, requested.arguments)
                steps.append(Step(kind="action", detail=requested.name, output=observation))
                messages.append(
                    Message(
                        role="tool",
                        content=observation,
                        tool_call_id=requested.id or requested.name,
                    )
                )

        # The cap is a control, not an excuse for an empty answer: one
        # tool-free call so the caller always gets something.
        self._logger.warning("ReAct hit its step cap", cap=self._max_steps)
        final = await self._llm.complete(CompletionRequest(messages=messages))
        steps.append(Step(kind="forced_answer", detail="cap reached", output=final.content))
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=final.content,
            steps=steps,
            metadata={"steps_taken": self._max_steps, "hit_cap": True},
        )

    async def _observe(self, name: str, arguments: str) -> str:
        """Run one requested action and return what it observed."""
        call = self._registry.build_call(name, arguments)
        if call is None:
            return f"'{name}' is unavailable or its arguments were invalid."
        result = await invoke_tool(self._registry, call)
        return result.output if result.success else f"failed: {result.error}"
