"""E14 — Building Each Store: core retrieval logic.

grep is a legitimate retrieval primitive, not a fallback for when "real"
retrieval is unavailable. The NOLLM cascade reapplied to lookup: exact
key, then a literal scan, then fuzzy matching, and only then an embedding.

For well-structured facts the first tier answers almost everything, which
is why reaching for embeddings first is both slower and worse.
"""

import tempfile
from pathlib import Path

from agentic_commons.memory import (
    EpisodicStore,
    MemoryFact,
    MemoryQuery,
    ProceduralStore,
    Provenance,
    SemanticStore,
)
from agentic_commons.persistence import IN_MEMORY, SqliteConnectionPool
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Exercise each store's retrieval, cheapest tier first."""
    output = EpisodeOutput(
        episode="E14",
        title="Building Each Store",
        principle="Exact match, then grep, then fuzzy, then embeddings — in that order.",
    )
    pool = SqliteConnectionPool(IN_MEMORY)
    semantic = SemanticStore(pool)
    episodic = EpisodicStore(Path(tempfile.mkdtemp()))
    procedural = ProceduralStore(pool)
    provenance = Provenance(source="policy", kind="verified_system_of_record")

    for key, value in [
        ("refund_window_days", "30"),
        ("damage_report_days", "14"),
        ("delivery_standard_days", "3-5"),
    ]:
        semantic.write(MemoryFact(key=key, value=value, provenance=provenance), provenance)

    output.say("[bold]Semantic — exact key first[/bold]")
    output.say(f"  get('refund_window_days')   → {semantic.get('refund_window_days')[0].value}")
    fuzzy = semantic.fuzzy_get("refund_window_day")
    output.say(
        f"  fuzzy('refund_window_day')  → {fuzzy[0].value if fuzzy else 'no match'}  (near-miss key)"
    )
    found = semantic.read(MemoryQuery(text="30"))
    output.say(f"  search('30')                → {len(found)} hit(s) by literal scan")

    for text in ["customer asked about order 55", "refund issued for order 55"]:
        episodic.write(
            MemoryFact(
                key="event",
                value=text,
                category="event",
                metadata={"session_id": "s1"},
                provenance=Provenance(source="agent"),
            ),
            Provenance(source="agent"),
        )
    grepped = episodic.grep("refund")
    output.say("")
    output.say("[bold]Episodic — one file per session, grep-able by construction[/bold]")
    output.say(f"  sessions on disk : {episodic.sessions()}")
    output.say(f"  grep('refund')   → {[f.value for f in grepped]}")

    for body in ["ask for the order id", "ask for the order id, then verify the email"]:
        procedural.write(
            MemoryFact(key="refund_howto", value=body, provenance=Provenance(source="curator")),
            Provenance(source="curator"),
        )
    active = procedural.get("refund_howto")
    procedural.rollback("refund_howto", 1)
    rolled = procedural.get("refund_howto")
    output.say("")
    output.say("[bold]Procedural — versioned, so a bad revision is reversible[/bold]")
    output.say(f"  versions   : {len(procedural.history('refund_howto'))}")
    output.say(f"  active     : {active.value if active else ''!r}")
    output.say(f"  rolled back: {rolled.value if rolled else ''!r}")

    return output.model_copy(
        update={
            "artifact": {
                "stores built": "semantic, episodic, procedural",
                "semantic tiers": "exact → LIKE scan → fuzzy key → embedding",
                "episodic": f"{len(episodic.sessions())} session file(s), grep-able",
                "procedural versions": len(procedural.history("refund_howto")),
            }
        }
    )


if __name__ == "__main__":
    main(run)
