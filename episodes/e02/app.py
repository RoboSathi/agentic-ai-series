"""E02 — Agency ≠ Authority.

An agent can *attempt* anything its tools allow — that is agency. What it
is *permitted* to do is authority, and the two are set by different
people at different times.

The demonstration is the structuring attack. Eighty small refunds are
eighty individually-permitted actions and one large loss. Every
per-call check passes; the sequence is the problem, and a checker that
sees one call at a time cannot see it by construction.
"""

from agent_platform.audit import SqliteAuditLedger
from agent_platform.gatekeeper import AuthorityContext, GatekeeperImpl
from agent_platform.gatekeeper.ledger import CumulativeLedger
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from agent_platform.security import Authorizer
from agent_platform.tools import ToolCall
from episodes._shared import EpisodeOutput, main

REFUND_EACH = 50.0


async def run() -> EpisodeOutput:
    """Run the structuring attack against the cumulative ledger."""
    output = EpisodeOutput(
        episode="E02",
        title="Agency ≠ Authority",
        principle="Every individual action was permitted. The sequence was not.",
    )
    pool = SqliteConnectionPool(IN_MEMORY)
    audit = SqliteAuditLedger(pool)
    # The ledger is on here whatever the deploying agent declares: this
    # episode demonstrates the platform mechanism, not one agent's posture.
    gatekeeper = GatekeeperImpl(
        authorizer=Authorizer(), ledger=CumulativeLedger(pool, enabled=True), audit=audit
    )
    authority = AuthorityContext(actor="agent-1", roles=["support_agent"], granted_by="user_task")

    allowed = denied = 0
    first_refusal = 0
    for attempt in range(1, 100):
        call = ToolCall(action="process_refund", params={"amount": REFUND_EACH})
        decision = await gatekeeper.evaluate(call, authority)
        if decision.verdict == "allow":
            allowed += 1
            gatekeeper.commit(call, authority)
        else:
            denied += 1
            first_refusal = first_refusal or attempt

    threshold = CumulativeLedger(pool).threshold_for("refund")
    output.say(f"Each refund is ${REFUND_EACH:.0f} — well within every per-call rule.")
    output.say(f"  allowed        : {allowed}  (${allowed * REFUND_EACH:,.0f})")
    output.say(f"  refused        : {denied}")
    output.say(f"  first refusal  : attempt #{first_refusal}")
    output.say("")
    output.say(f"The ledger holds the memory a per-call check lacks: ${threshold:,.0f}")
    output.say("cumulative, in a rolling window, per actor.")
    output.say("")
    output.say(f"Every decision is on the record: {audit.count()} audit entries,")
    output.say(f"{len(audit.by_decision('deny'))} of them refusals with a stated reason.")

    return output.model_copy(
        update={
            "artifact": {
                "per-call verdict": "allow (always)",
                "cumulative threshold": f"${threshold:,.0f}",
                "attempts allowed": allowed,
                "attempts refused": denied,
                "loss prevented": f"${denied * REFUND_EACH:,.0f}",
                "audit entries": audit.count(),
            }
        }
    )


if __name__ == "__main__":
    main(run)
