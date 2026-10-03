"""Tool use — the model chooses the call (E05).

The smallest step from workflow to agent: a person still writes every
tool, but the model decides which to call and with what arguments.

What changes is not capability but *accountability*. Once the model picks
the action, "what can this system do?" stops being answerable by reading
the code and becomes a question about what the model might choose. That
is why the harness validates arguments before execution and why a
Gatekeeper exists at all — both are answers to a question a workflow
never has to ask.

Composes `agent_platform.tools`: the registry owns what exists and
validates the call; this pattern owns only the loop shape.
"""

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import get_logger
from agent_platform.tools import NamespacedToolRegistry
from design_patterns.agentic._invoke import invoke_tool
from design_patterns.types import PatternFamily, PatternRun, Step


class ToolUsePattern:
    """One model turn that may call a tool, then answer from the result."""

    name: str = "tool_use"
    family: PatternFamily = "AG"

    def __init__(self, llm: LLMClient, registry: NamespacedToolRegistry) -> None:
        self._llm = llm
        self._registry = registry
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Offer the tools, run whatever the model asks for, then answer."""
        messages = [Message(role="user", content=task)]
        first = await self._llm.complete(
            CompletionRequest(messages=messages, tools=self._registry.to_openai_tools())
        )
        steps = [
            Step(
                kind="model", detail="chose", output=first.content, tokens=first.usage.total_tokens
            )
        ]

        if not first.tool_calls:
            return PatternRun(
                pattern=self.name, family=self.family, output=first.content, steps=steps
            )

        messages.append(
            Message(role="assistant", content=first.content, tool_calls=first.tool_calls)
        )
        for requested in first.tool_calls:
            call = self._registry.build_call(requested.name, requested.arguments)
            if call is None:
                # The model's error to see and recover from, not the
                # harness's to crash on.
                observation = f"'{requested.name}' is unavailable or its arguments were invalid."
            else:
                result = await invoke_tool(self._registry, call)
                observation = result.output if result.success else f"failed: {result.error}"
            steps.append(Step(kind="tool", detail=requested.name, output=observation))
            messages.append(
                Message(
                    role="tool", content=observation, tool_call_id=requested.id or requested.name
                )
            )

        final = await self._llm.complete(CompletionRequest(messages=messages))
        steps.append(
            Step(
                kind="model",
                detail="answered",
                output=final.content,
                tokens=final.usage.total_tokens,
            )
        )
        return PatternRun(pattern=self.name, family=self.family, output=final.content, steps=steps)
