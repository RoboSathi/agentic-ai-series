"""E33 — Data Lineage & Regulator-Ready Evidence Chains.

An audit trail answers a regulator's question, not an engineer's: who was
allowed to do this, under what authority, against which version of the
model, the policy and the data?

Anything less is a log file. The distinguishing property is
attributability to a versioned tuple, and reconstruction in *causal*
order — insert sequence, not wall clock, because two decisions inside the
same millisecond still happened in an order.

Append-only is enforced by construction: the ledger exposes `record` and
`reconstruct` and has no method that issues UPDATE or DELETE.
"""

from agent_platform.audit import AuditEntry, SqliteAuditLedger
from agent_platform.gatekeeper import AuthorityContext, GatekeeperImpl
from agent_platform.gatekeeper.ledger import CumulativeLedger
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from agent_platform.security import Authorizer
from agent_platform.tools import ToolCall
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Produce a trace, reconstruct it, and show the append-only property."""
    output = EpisodeOutput(
        episode="E33",
        title="Data Lineage & Evidence Chains",
        principle="Attributable to a versioned (model, policy, data, authority) tuple — or it is a log file.",
    )
    pool = SqliteConnectionPool(IN_MEMORY)
    audit = SqliteAuditLedger(pool)
    gatekeeper = GatekeeperImpl(authorizer=Authorizer(), ledger=CumulativeLedger(pool), audit=audit)
    authority = AuthorityContext(actor="agent-1", roles=["support_agent"], granted_by="user_task")

    for action, params in [
        ("get_order_status", {}),
        ("process_refund", {"amount": 20.0}),
        ("update_refund_policy", {}),
    ]:
        await gatekeeper.evaluate(ToolCall(action=action, params=params), authority)

    audit.record(
        AuditEntry(
            trace_id="", actor="agent-1", action="manual", decision="allow", rationale="direct"
        )
    )
    entries = audit.reconstruct("")

    output.say(f"[bold]{audit.count()} decisions recorded[/bold]")
    for entry in audit.by_decision("allow") + audit.by_decision("deny"):
        output.say(f"  {entry.decision:<9} {entry.action:<22} {entry.rationale[:40]}")

    sample = (audit.by_decision("allow") or entries)[0]
    output.say("")
    output.say("[bold]One entry, in full[/bold]")
    output.say(f"  actor          : {sample.actor}")
    output.say(f"  model version  : {sample.model_version}")
    output.say(f"  policy version : {sample.policy_version}")
    output.say(f"  data version   : {sample.data_version}")
    output.say(
        f"  authority      : {sample.authority_context.get('granted_by')} (rank {sample.authority_context.get('granted_by_rank')})"
    )

    forbidden = {"update", "delete", "edit", "amend", "remove"}
    surface = set(dir(SqliteAuditLedger))
    output.say("")
    output.say("[bold]Append-only by construction[/bold]")
    output.say(f"  mutating methods on the ledger: {sorted(forbidden & surface) or 'none'}")
    output.say("  There is nothing to call, so there is nothing to review for")
    output.say("  having been called.")

    return output.model_copy(
        update={
            "artifact": {
                "entries": audit.count(),
                "versioned tuple": "model, policy, data, authority",
                "ordering": "insert sequence (causal), not wall clock",
                "mutating methods": len(forbidden & surface),
                "denials on record": len(audit.by_decision("deny")),
            }
        }
    )


if __name__ == "__main__":
    main(run)
