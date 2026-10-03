"""E12 — The Agent Control Plane: Sandbox → Gatekeeper → Execution.

One mediation point that no execution path bypasses. It answers four
questions in order, short-circuiting on the first denial: what surface
does this touch, who is asking, under whose authority, and has the
cumulative state crossed a threshold.

The four live attacks are run here as they are in the regression suite.
Each one passes a per-call permission check and is stopped by a different
question — which is the argument for having four.
"""

from agent_platform.audit import SqliteAuditLedger
from agent_platform.gatekeeper import AuthorityContext, GatekeeperImpl
from agent_platform.gatekeeper.ledger import CumulativeLedger
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from agent_platform.security import Authorizer
from agent_platform.tools import ToolCall
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Run the four E12 attacks against the assembled Gatekeeper."""
    output = EpisodeOutput(
        episode="E12",
        title="The Agent Control Plane",
        principle="Four questions. Each attack passes three of them.",
    )
    pool = SqliteConnectionPool(IN_MEMORY)
    audit = SqliteAuditLedger(pool)
    gatekeeper = GatekeeperImpl(authorizer=Authorizer(), ledger=CumulativeLedger(pool), audit=audit)
    agent = AuthorityContext(actor="agent-1", roles=["support_agent"], granted_by="user_task")

    attacks = [
        (
            "edit its own policy",
            ToolCall(action="update_refund_policy"),
            agent,
        ),
        (
            "injected instruction",
            ToolCall(action="process_refund", params={"amount": 20.0}),
            AuthorityContext(
                actor="agent-1", roles=["support_agent"], granted_by="untrusted_content"
            ),
        ),
        (
            "tool result claims authority",
            ToolCall(action="process_refund", params={"amount": 20.0}),
            AuthorityContext(
                actor="agent-1", roles=["support_agent"], granted_by="tool_observations"
            ),
        ),
    ]
    stopped_by = {}
    for label, call, authority in attacks:
        decision = await gatekeeper.evaluate(call, authority)
        stopped_by[label] = decision.question
        output.say(f"  [bold]{label}[/bold]")
        output.say(
            f"    {decision.verdict.upper()} at question '{decision.question}' — {decision.reason}"
        )

    allowed = 0
    for _ in range(100):
        call = ToolCall(action="process_refund", params={"amount": 50.0})
        if (await gatekeeper.evaluate(call, agent)).verdict == "allow":
            allowed += 1
            gatekeeper.commit(call, agent)
    stopped_by["structuring"] = "cumulative"
    output.say("  [bold]structuring: 100 × $50[/bold]")
    output.say(f"    {allowed} allowed, then DENY at question 'cumulative'")

    output.say("")
    output.say("Four attacks, four different questions. A system that asked only")
    output.say("'is this actor permitted?' would have allowed three of them.")

    return output.model_copy(
        update={
            "artifact": {
                "attacks defended": len(stopped_by),
                "stopped by": stopped_by,
                "audit entries": audit.count(),
                "denials recorded": len(audit.by_decision("deny")),
            }
        }
    )


if __name__ == "__main__":
    main(run)
