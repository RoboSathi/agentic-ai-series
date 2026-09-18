"""E13 — Memory Architecture: working, semantic, episodic, procedural.

CoALA's four stores. The central rule: which store a fact belongs in is a
semantic decision about the fact's nature, never a storage convenience.

Retrieval is composite — Park et al.'s relevance + recency + importance,
extended with authority and temporal validity, because once memory is a
security boundary a self-reported claim must not outrank a system of
record and an expired fact must stop winning.
"""

from datetime import timedelta

from agentic_commons.memory import (
    CompositeRetrievalScorer,
    EpisodicStore,
    MemoryFact,
    ProceduralStore,
    Provenance,
    SemanticStore,
    WorkingMemory,
)
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import IN_MEMORY, SqliteConnectionPool
from episodes._shared import EpisodeOutput, main

STORES = {
    "working": "what is active now — the context window itself, not a store",
    "semantic": "what is generally true — structured facts",
    "episodic": "what happened — append-only session logs",
    "procedural": "how to do things — versioned how-to artifacts",
}


async def run() -> EpisodeOutput:
    """Show the four stores and why the composite score needs five signals."""
    import tempfile
    from pathlib import Path

    output = EpisodeOutput(
        episode="E13",
        title="Memory Architecture",
        principle="Which store a fact belongs in is semantic, not a storage convenience.",
    )
    pool = SqliteConnectionPool(IN_MEMORY)
    tmp = Path(tempfile.mkdtemp())
    stores = {
        "working": WorkingMemory(),
        "semantic": SemanticStore(pool),
        "episodic": EpisodicStore(tmp),
        "procedural": ProceduralStore(pool),
    }

    output.say("[bold]The four CoALA stores[/bold]")
    for name, description in STORES.items():
        output.say(f"  {name:<12} {description}")

    output.say("")
    output.say("[bold]Why one signal is not enough[/bold]")
    scorer = CompositeRetrievalScorer()

    verified = MemoryFact(
        key="refund_limit",
        value="50",
        provenance=Provenance(source="policy", kind="verified_system_of_record"),
    )
    claimed = MemoryFact(
        key="refund_limit",
        value="5000",
        provenance=Provenance(source="customer", kind="self_reported"),
    )
    superseded = verified.model_copy(update={"superseded_by": "refund_limit_v2"})
    stale = verified.model_copy(update={"valid_until": utc_now() - timedelta(days=1)})

    for label, fact in [
        ("verified fact", verified),
        ("self-reported claim", claimed),
        ("superseded fact", superseded),
        ("expired fact", stale),
    ]:
        scored = scorer.score(fact, relevance=1.0)
        output.say(
            f"  {label:<20} score {scored.score:.2f}  "
            f"(authority {scored.authority:.2f}, validity {scored.temporal_validity:.0f})"
        )

    output.say("")
    output.say("All four are equally *relevant*. Only the composite separates them —")
    output.say("which is what stops a superseded fact winning on similarity alone.")

    return output.model_copy(
        update={
            "artifact": {
                "stores": ", ".join(STORES),
                "all implemented": all(store is not None for store in stores.values()),
                "signals": "relevance, recency, importance, authority, temporal validity",
                "verified vs claimed": f"{scorer.score(verified, 1.0).score:.2f} vs {scorer.score(claimed, 1.0).score:.2f}",
            }
        }
    )


if __name__ == "__main__":
    main(run)
