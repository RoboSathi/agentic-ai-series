"""E16 — Memory Is a Security Boundary: poisoning, revocation, false recall.

Memory poisoning needs no hidden text. A user can simply *say* a false
thing, intending it to be trusted later.

Three defences, each tested here: provenance weighting so a self-reported
claim never outranks a system of record; decay-if-unconfirmed so an
unverified claim loses standing rather than hardening into fact through
persistence; and hard revocation, which actually deletes — a "deletion"
that leaves the value readable is a deletion failure.

The exit eval targets *false recall* — confidently retrieving the wrong
stored fact — which is a different failure from clean forgetting.
"""

import tempfile
from datetime import timedelta
from pathlib import Path

from agent_platform.memory import (
    EpisodicStore,
    MemoryFact,
    MemoryQuery,
    Provenance,
    SemanticStore,
)
from agent_platform.memory.facade import MemoryFacade
from agent_platform.observability.types import utc_now
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Poison, supersede, decay and revoke — then check recall."""
    output = EpisodeOutput(
        episode="E16",
        title="Memory Is a Security Boundary",
        principle="False recall is a false positive. It is not the same failure as forgetting.",
    )
    pool = SqliteConnectionPool(IN_MEMORY)
    semantic = SemanticStore(pool)
    facade = MemoryFacade.complete(semantic=semantic, episodic=EpisodicStore(Path(tempfile.mkdtemp())))
    verified = Provenance(source="policy", kind="verified_system_of_record")

    semantic.write(MemoryFact(key="refund_limit", value="50", provenance=verified), verified)
    semantic.supersede("refund_limit", "refund_limit_v2")
    semantic.write(MemoryFact(key="refund_limit_v2", value="75", provenance=verified), verified)

    recalled = await facade.recall(MemoryQuery(text="refund_limit", limit=5))
    output.say("[bold]False recall[/bold] — a superseded fact still matches on similarity")
    for scored in recalled:
        output.say(
            f"  {scored.fact.key:<18} = {scored.fact.value:<5} "
            f"score {scored.score:.2f}  validity {scored.temporal_validity:.0f}"
        )
    output.say("  The superseded fact scores zero validity, so it cannot win.")

    stale = MemoryFact(
        key="claimed_discount",
        value="40%",
        provenance=Provenance(source="customer", kind="self_reported"),
    )
    stale.provenance.ingestion_time = utc_now() - timedelta(days=400)
    semantic.write(stale, stale.provenance)
    expired = semantic.decay_unconfirmed()
    output.say("")
    output.say("[bold]Decay-if-unconfirmed[/bold] — a security posture, not hygiene")
    output.say(f"  unconfirmed self-reported facts expired: {expired}")

    semantic.write(
        MemoryFact(key="customer_email", value="ada@example.com", provenance=verified), verified
    )
    removed = facade.forget("customer_email")
    remaining = semantic.get("customer_email", include_superseded=True)
    output.say("")
    output.say("[bold]Revocation[/bold] — deletes, rather than annotating")
    output.say(f"  rows removed: {removed}, readable afterwards: {len(remaining)}")

    return output.model_copy(
        update={
            "artifact": {
                "threats": "poisoning, stale memory, false recall, authority contamination, deletion failure",
                "superseded recalled": any(s.fact.superseded_by for s in recalled),
                "decayed": expired,
                "revoked rows": removed,
                "readable after revoke": len(remaining),
            }
        }
    )


if __name__ == "__main__":
    main(run)
