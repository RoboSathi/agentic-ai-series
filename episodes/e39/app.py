"""E39 — Shipping Agent Changes Safely: version everything that moves.

Four things move independently, and a change to any of them can alter
behaviour: the model, the prompt, the policy, and the data. A deployment
that versions only the code can tell you what shipped and not what
changed.

Every audit entry already carries three of the four, which is what makes
"why did it behave differently last Tuesday?" answerable at all.
"""

from agent_platform.audit import AuditEntry, SqliteAuditLedger
from agent_platform.config import get_settings
from agent_platform.memory import MemoryFact, ProceduralStore, Provenance
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Show what is versioned, and roll a prompt back."""
    output = EpisodeOutput(
        episode="E39",
        title="Shipping Agent Changes Safely",
        principle="Four things move independently. Versioning the code versions one of them.",
    )
    settings = get_settings()
    pool = SqliteConnectionPool(IN_MEMORY)

    moving = {
        "model": settings.llm.providers[settings.llm.provider].model,
        "policy": settings.audit.policy_version,
        "data": settings.audit.data_version,
        "prompt": "versioned in memory.ProceduralStore",
    }
    output.say("[bold]Four independently moving parts[/bold]")
    for part, version in moving.items():
        output.say(f"  {part:<8} {version}")

    audit = SqliteAuditLedger(pool)
    audit.record(
        AuditEntry(
            trace_id="t1",
            actor="agent-1",
            action="process_refund",
            decision="allow",
            model_version=moving["model"],
            rationale="within threshold",
        )
    )
    entry = audit.reconstruct("t1")[0]
    output.say("")
    output.say("[bold]Every decision carries them[/bold]")
    output.say(
        f"  model {entry.model_version} · policy {entry.policy_version} · data {entry.data_version}"
    )

    store = ProceduralStore(pool)
    curator = Provenance(source="release", kind="verified_system_of_record")
    for body in ["v1: ask for the order id", "v2: ask for the order id and verify the email"]:
        store.write(MemoryFact(key="system_prompt", value=body, provenance=curator), curator)
    before = store.get("system_prompt")
    store.rollback("system_prompt", 1)
    after = store.get("system_prompt")

    output.say("")
    output.say("[bold]Rollback[/bold] — the property that makes shipping safe")
    output.say(f"  shipped  : {before.value if before else ''}")
    output.say(f"  rolled to: {after.value if after else ''}")
    output.say("  A prompt edited in place cannot be rolled back when the")
    output.say("  'improvement' turns out to be a regression.")

    return output.model_copy(
        update={
            "artifact": {
                "versioned": ", ".join(moving),
                "model": moving["model"],
                "policy": moving["policy"],
                "data": moving["data"],
                "prompt versions": len(store.history("system_prompt")),
                "rollback works": (after.metadata["version"] if after else 0) == 1,
            }
        }
    )


if __name__ == "__main__":
    main(run)
