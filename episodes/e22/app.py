"""E22 — Tool Selection at Scale: registry, namespacing, retrieval.

Exposing every tool does not scale. Past a few dozen, selection accuracy
falls even though nothing about any individual tool changed — the model
is choosing from a longer list, and the list itself is the problem.

Two halves of the recovery: namespacing scopes what is visible, retrieval
ranks what remains. The four exit metrics are measured here, not asserted:
token budget, tools exposed, irrelevant exposure rate, latency.
"""

from pydantic import BaseModel

from agentic_commons.evals import create_scoring_cascade
from agentic_commons.tools import (
    NamespacedToolRegistry,
    ToolDefinition,
    ToolResult,
    ToolRetrievalSelector,
)
from episodes._shared import EpisodeOutput, main


class Args(BaseModel):
    value: str = ""


def _noop(**kwargs: object) -> ToolResult:
    return ToolResult(success=True, output="ok")


def _registry(count: int) -> NamespacedToolRegistry:
    """A registry with one relevant tool buried among many."""
    registry = NamespacedToolRegistry()
    for index in range(count):
        registry.register(
            ToolDefinition(
                name=f"task_{index}",
                namespace="ops",
                description=f"Perform routine operations task number {index}",
                args_schema=Args,
                handler=_noop,
            )
        )
    registry.register(
        ToolDefinition(
            name="process_refund",
            namespace="billing",
            description="Issue a refund to a customer for an order",
            args_schema=Args,
            handler=_noop,
        )
    )
    return registry


async def run() -> EpisodeOutput:
    """Measure unfiltered exposure against namespaced retrieval."""
    output = EpisodeOutput(
        episode="E22",
        title="Tool Selection at Scale",
        principle="Namespacing scopes what is visible. Retrieval ranks what remains.",
    )
    registry = _registry(50)
    selector = ToolRetrievalSelector(registry, create_scoring_cascade())
    task = "refund the customer their money for this order"

    unfiltered = registry.to_openai_tools()
    unfiltered_chars = sum(len(str(schema)) for schema in unfiltered)
    output.say(
        f"[bold]Unfiltered[/bold]: {len(unfiltered)} tools, ~{unfiltered_chars // 4:,} tokens of schema"
    )
    output.say("  The model picks from 51 near-identical descriptions.")

    selected = await selector.select_scored(task, k=5)
    metrics = selector.last_metrics
    output.say("")
    output.say(f"[bold]After retrieval[/bold]: {len(selected)} exposed")
    for choice in selected:
        output.say(f"  {choice.score:.3f}  {choice.tool}")

    assert metrics is not None
    output.say("")
    output.say("[bold]The four exit metrics[/bold]")
    output.say(f"  tools available        : {metrics.tools_available}")
    output.say(f"  tools exposed          : {metrics.tools_exposed}")
    output.say(f"  exposed token estimate : {metrics.exposed_token_estimate:,}")
    output.say(f"  irrelevant exposure    : {metrics.irrelevant_exposure_rate:.1%}")
    output.say(f"  selection latency      : {metrics.latency_ms:.0f}ms")

    scoped = registry.in_namespace("billing")
    output.say("")
    output.say(
        f"[bold]Namespacing alone[/bold]: {len(scoped)} tool(s) in 'billing' vs {len(unfiltered)} overall"
    )
    top = selected[0].tool if selected else "nothing"
    output.say(f"  correct tool ranked first: {top == 'billing.process_refund'}")

    return output.model_copy(
        update={
            "artifact": {
                "tools available": metrics.tools_available,
                "tools exposed": metrics.tools_exposed,
                "token budget before": unfiltered_chars // 4,
                "token budget after": metrics.exposed_token_estimate,
                "irrelevant exposure rate": metrics.irrelevant_exposure_rate,
                "latency ms": round(metrics.latency_ms),
            }
        }
    )


if __name__ == "__main__":
    main(run)
