"""E05 — Tool Use & ReAct: the foundational loop.

Yao et al. (2022): interleaving reasoning with actions beats either alone.
The demonstration is a task a single completion cannot finish — it needs
two lookups, and the second depends on the first.

Exit artifact: a trajectory of thought/action/observation triples.
"""

from pydantic import BaseModel

from agentic_commons.tools import NamespacedToolRegistry, ToolDefinition, ToolResult
from design_patterns import ReActPattern, ToolUsePattern
from episodes._shared import EpisodeOutput, OfflineLLM, main

FACTS = {
    "order_55": "Order 55: delivered 3 March, total $120.00",
    "policy": "Refunds are available for 30 days after delivery",
}


class LookupArgs(BaseModel):
    key: str


def _registry() -> NamespacedToolRegistry:
    def lookup(key: str) -> ToolResult:
        value = FACTS.get(key)
        return (
            ToolResult(success=True, output=value)
            if value
            else ToolResult(success=False, error=f"no entry for {key}")
        )

    registry = NamespacedToolRegistry()
    registry.register(
        ToolDefinition(
            name="lookup",
            namespace="kb",
            description="Look up a fact by key",
            args_schema=LookupArgs,
            handler=lookup,
        )
    )
    return registry


async def run() -> EpisodeOutput:
    """Contrast one completion with a ReAct trajectory on the same task."""
    import json

    from agentic_commons.llm.types import ModelToolCall

    output = EpisodeOutput(
        episode="E05",
        title="Tool Use & ReAct",
        principle="Reason → act → observe. The trajectory is the unit of work, and the audit record.",
    )
    registry = _registry()
    task = "Can order 55 still be refunded?"

    single = OfflineLLM(["Probably, most orders can be refunded."])
    once = await ToolUsePattern(single, registry).run(task)
    output.say("[bold]One completion, no tools[/bold]")
    output.say(f"  → {once.output!r}")
    output.say("  Plausible, unsourced, and unfalsifiable. It looked nothing up.")

    class Scripted(OfflineLLM):
        def __init__(self) -> None:
            super().__init__()
            self._turns = [
                ("I need the order.", [("kb__lookup", {"key": "order_55"})]),
                ("Now the policy.", [("kb__lookup", {"key": "policy"})]),
                ("Delivered 3 March; the 30-day window has passed.", []),
            ]
            self._at = 0

        async def complete(self, request):  # type: ignore[no-untyped-def]
            from agentic_commons.llm.types import CompletionResult, TokenUsage

            content, calls = self._turns[min(self._at, len(self._turns) - 1)]
            self._at += 1
            return CompletionResult(
                content=content,
                tool_calls=[ModelToolCall(id=n, name=n, arguments=json.dumps(a)) for n, a in calls]
                or None,
                usage=TokenUsage(total_tokens=40),
                tier="offline",
            )

    trajectory = await ReActPattern(Scripted(), registry, max_steps=5).run(task)
    output.say("")
    output.say("[bold]The same task as a ReAct loop[/bold]")
    for step in trajectory.steps:
        output.say(f"  {step.kind:<8} {step.detail:<10} {step.output[:60]}")
    output.say("")
    output.say(f"  → {trajectory.output!r}")
    output.say("  Every claim traces to something it actually read.")

    return output.model_copy(
        update={
            "artifact": {
                "single completion": "1 step, 0 observations",
                "react trajectory": f"{trajectory.step_count} steps",
                "thoughts": len(trajectory.steps_of("thought")),
                "actions": len(trajectory.steps_of("action")),
                "hit cap": trajectory.metadata.get("hit_cap"),
            }
        }
    )


if __name__ == "__main__":
    main(run)
