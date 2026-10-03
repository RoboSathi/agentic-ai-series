"""E18 — Context Management: offload, reduce, retrieve, isolate, cache, consolidate.

Six strategies. They are not interchangeable — each trades something
different away, and choosing one means naming what you are willing to
lose.

They are applied cheapest-loss first: deduplicate, drop the least
relevant, offload bulk, and only then reduce — because reduction is the
only one of the four that destroys information outright.
"""

from agent_platform.context import (
    AssembledContext,
    CacheStrategy,
    Chunk,
    ConsolidateStrategy,
    IsolateStrategy,
    OffloadStrategy,
    ReduceStrategy,
    RetrieveStrategy,
    StrategySelector,
)
from episodes._shared import EpisodeOutput, main

TRADES = {
    "offload": "loses immediacy — the agent must fetch it back",
    "reduce": "loses detail, permanently",
    "retrieve": "loses whatever you failed to ask for",
    "isolate": "costs a round trip",
    "cache": "loses the ability to vary the prefix",
    "consolidate": "loses the fact that something was repeated",
}


async def run() -> EpisodeOutput:
    """Apply each strategy, then let the selector fit an oversized window."""
    output = EpisodeOutput(
        episode="E18",
        title="Context Management",
        principle="Each strategy trades something different away. Name what you are losing.",
    )

    output.say("[bold]Six strategies, six different costs[/bold]")
    for name, trade in TRADES.items():
        output.say(f"  {name:<12} {trade}")

    oversized = AssembledContext(
        chunks=[
            Chunk(
                content="you are a support agent",
                kind="system",
                source="sys",
                pinned=True,
                token_estimate=8,
            ),
            Chunk(
                content="the refund policy is thirty days",
                kind="policy",
                source="policy",
                token_estimate=40,
            ),
            Chunk(
                content="the refund policy is thirty days",
                kind="policy",
                source="policy",
                token_estimate=40,
            ),
            Chunk(
                content="x" * 2000, kind="retrieved", source="manual", token_estimate=500, score=0.1
            ),
            Chunk(
                content="order 55 delivered March 3",
                kind="retrieved",
                source="orders",
                token_estimate=20,
                score=0.9,
            ),
        ],
        budget_tokens=80,
        used_tokens=608,
    )
    output.say("")
    output.say(
        f"[bold]An oversized window[/bold]: {oversized.used_tokens} tokens against a {oversized.budget_tokens} budget"
    )

    for strategy in (
        ConsolidateStrategy(),
        RetrieveStrategy(top_k=2),
        OffloadStrategy(min_tokens=100),
        ReduceStrategy(keep_ratio=0.5),
    ):
        applied = strategy.apply(oversized)
        output.say(f"  {strategy.name:<12} → {applied.used_tokens:>4} tokens")

    fitted = StrategySelector().fit(oversized)
    output.say("")
    output.say(f"[bold]The selector[/bold] stops as soon as it fits: {fitted.used_tokens} tokens")
    output.say(f"  applied in order: {' → '.join(fitted.applied_strategies)}")
    output.say(f"  the pinned system prompt survived: {any(c.pinned for c in fitted.chunks)}")

    cached = CacheStrategy()
    prefix = cached.prefix_signature(cached.apply(oversized))
    isolated = IsolateStrategy().partition(oversized)
    output.say("")
    output.say(
        f"[bold]Cache[/bold]: a stable prefix of {len(prefix)} chars, ordered deterministically"
    )
    output.say(f"[bold]Isolate[/bold]: partitioned into {len(isolated)} windows by kind")

    return output.model_copy(
        update={
            "artifact": {
                "strategies": ", ".join(TRADES),
                "before": f"{oversized.used_tokens} tokens",
                "after": f"{fitted.used_tokens} tokens",
                "applied": " → ".join(fitted.applied_strategies),
                "pinned survived": any(c.pinned for c in fitted.chunks),
            }
        }
    )


if __name__ == "__main__":
    main(run)
