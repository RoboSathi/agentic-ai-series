"""E36 — Observability for Agents: trace the decision, not just the call.

A trace that records "the model was called, it took 900ms" is telemetry.
A trace that can answer *why* an action was permitted is audit.

The difference is what a span carries: actor, authority context, policy
version, model version, the state transition, and the result — enough to
reconstruct a decision, not merely to confirm a call occurred.

And PII is redacted before persistence, because a trace outlives the
request that produced it.
"""

from agentic_commons.observability import LocalTracer
from agentic_commons.persistence import IN_MEMORY, SqliteConnectionPool
from agentic_commons.security import PIIFilterImpl
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Record a decision-carrying span and show redaction before storage."""
    output = EpisodeOutput(
        episode="E36",
        title="Observability for Agents",
        principle="A trace that cannot answer 'why' is telemetry, not audit.",
    )
    pool = SqliteConnectionPool(IN_MEMORY)
    tracer = LocalTracer(pool, redactor=PIIFilterImpl())

    with tracer.span(
        "refund_decision",
        actor="agent-1",
        tool="process_refund",
        policy_version="refund-policy-v1",
        model_version="nemotron-3",
        state_transition="ACTING → WAITING_HUMAN",
        result="escalated: ada@example.com asked for $210",
        cost_usd=0.0004,
    ) as span:
        trace_id = span.trace_id
        with tracer.span("order_lookup", parent=span, actor="agent-1", tool="get_order_status"):
            pass

    spans = tracer.spans_for(trace_id)
    output.say(f"[bold]{len(spans)} spans in trace {trace_id[:12]}…[/bold]")
    for recorded in spans:
        parent = "root" if recorded.parent_id is None else "child"
        output.say(
            f"  {parent:<6} {recorded.name:<18} {recorded.latency_ms:.1f}ms  actor={recorded.actor}"
        )

    root = next(s for s in spans if s.parent_id is None)
    output.say("")
    output.say("[bold]What the span carries[/bold] — enough to reconstruct the decision")
    for field in ("actor", "tool", "policy_version", "model_version", "state_transition"):
        output.say(f"  {field:<18} {getattr(root, field)}")

    stored = pool.query_one("SELECT result FROM trace_spans WHERE trace_id = ?", (trace_id,))
    output.say("")
    output.say("[bold]Redacted before persistence[/bold]")
    output.say(f"  in memory : {root.result}")
    output.say(f"  on disk   : {stored['result'] if stored else ''}")
    output.say("  The live view keeps the text an operator needs. What outlives")
    output.say("  the request does not.")

    return output.model_copy(
        update={
            "artifact": {
                "spans": len(spans),
                "carries": "actor, authority, tool, policy version, model version, transition, result",
                "pii on disk": "redacted",
                "cost attributed": root.cost_usd,
            }
        }
    )


if __name__ == "__main__":
    main(run)
