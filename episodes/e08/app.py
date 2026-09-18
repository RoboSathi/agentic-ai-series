"""E08 — The Harness Contract: six subsystems, explicit guarantees.

Parnas's information hiding, applied to an agent harness: each subsystem
publishes an interface and hides how it keeps its promise, so a caller
cannot come to depend on the implementation.

The demonstration is structural. Every subsystem is fronted by a Protocol
with at least one alternative implementation, which is the only real test
of whether a boundary exists: if there is exactly one way to satisfy an
interface, it is a class, not a contract.
"""

from typing import Any

from agentic_commons.audit import AuditLedger, NullAuditLedger, SqliteAuditLedger
from agentic_commons.evals import NollmScoringCascade
from agentic_commons.llm.client import LLMClient
from agentic_commons.memory import MemoryStore
from agentic_commons.observability import LiveStatusFeed
from agentic_commons.persistence import Repository
from agentic_commons.tools import ToolRegistry
from episodes._shared import EpisodeOutput, main

CONTRACTS: list[tuple[str, Any, str]] = [
    ("llm", LLMClient, "complete / health_check"),
    ("persistence", Repository, "get / save / query"),
    ("tools", ToolRegistry, "register / get / all_tools"),
    ("memory", MemoryStore, "read / write"),
    ("audit", AuditLedger, "record / reconstruct"),
    ("observability", LiveStatusFeed, "emit / events_for"),
    ("evals", NollmScoringCascade, "resolve / nollm_scorer_ratio"),
]


async def run() -> EpisodeOutput:
    """Show each subsystem's interface and prove the boundary is real."""
    output = EpisodeOutput(
        episode="E08",
        title="The Harness Contract",
        principle="If there is only one way to satisfy an interface, it is a class, not a contract.",
    )

    output.say("[bold]Each subsystem publishes an interface, not an implementation[/bold]")
    for name, protocol, methods in CONTRACTS:
        output.say(f"  {name:<14} {protocol.__name__:<22} {methods}")

    output.say("")
    output.say("[bold]The boundary is real, not decorative[/bold]")
    real: AuditLedger = SqliteAuditLedger.__new__(SqliteAuditLedger)
    null: AuditLedger = NullAuditLedger()
    output.say(f"  AuditLedger is satisfied by {type(real).__name__} and {type(null).__name__}")
    output.say("  A caller that depended on SQLite could not accept the null one.")
    output.say("")
    output.say("Swapping an implementation is a constructor argument, not a refactor.")
    output.say("That property is what the whole series builds on.")

    return output.model_copy(
        update={
            "artifact": {
                "subsystems under contract": len(CONTRACTS),
                "interfaces": ", ".join(name for name, _, _ in CONTRACTS),
                "verified by": "mypy --strict, plus a null implementation per contract",
            }
        }
    )


if __name__ == "__main__":
    main(run)
