"""E38 — Human Review Is a Queue: Little's Law.

L = λW. Escalating more work to humans is a staffing decision with a
number attached, and the number is usually larger than anyone expects.

At 100 escalations an hour needing two minutes each, that is 3.33
reviewer-hours per hour — more than three full-time reviewers just to
break even. Below that staffing the queue does not get slower; it grows
without bound.

Which reframes "escalate when unsure" as a budget, not a safety net.
"""

from agentic_commons.observability import LittlesLawQueueModel
from agentic_commons.persistence import IN_MEMORY, SqliteConnectionPool
from customer_support_agent.core import HitlQueueRepository, OrdersRepository, ReviewQueueService
from customer_support_agent.core.refund_saga import RefundSaga
from customer_support_agent.core.repositories import RefundLedgerRepository
from customer_support_agent.core.schema import ensure_schema
from episodes._shared import EpisodeOutput, main

SCENARIOS = [
    (10, 2.0, "a small team"),
    (100, 2.0, "the E38 worked example"),
    (100, 8.0, "same volume, a harder review"),
    (1000, 2.0, "what 'escalate when unsure' looks like at scale"),
]


async def run() -> EpisodeOutput:
    """Compute staffing for several escalation rates, then drain a real queue."""
    output = EpisodeOutput(
        episode="E38",
        title="Human Review Is a Queue",
        principle="Escalating more is a staffing decision with a number attached.",
    )
    model = LittlesLawQueueModel()

    output.say(f"  {'arrivals/hr':>12} {'mins each':>10} {'reviewer-hrs/hr':>16} {'people':>7}")
    for arrivals, minutes, note in SCENARIOS:
        capacity = model.capacity_for(arrivals, minutes)
        output.say(
            f"  {arrivals:>12} {minutes:>10.1f} {capacity.reviewer_hours_per_hour:>16.2f} "
            f"{capacity.reviewers_required:>7}   {note}"
        )

    output.say("")
    output.say("[bold]Stability[/bold] — below the line the queue grows without bound")
    for staffing in (3, 4):
        stable = model.is_stable(100, 2.0, staffing)
        output.say(
            f"  100/hr, 2 min each, {staffing} reviewers → {'stable' if stable else 'UNSTABLE'}"
        )

    pool = SqliteConnectionPool(IN_MEMORY)
    ensure_schema(pool)
    orders = OrdersRepository(pool)
    service = ReviewQueueService(
        HitlQueueRepository(pool), orders, RefundSaga(orders, RefundLedgerRepository(pool))
    )
    output.say("")
    output.say("[bold]The queue drains[/bold] — an escalation nothing drains is not a control")
    output.say(f"  pending now : {service.pending_count()}")
    output.say("  approve() and deny() are operator entry points, deliberately not")
    output.say("  agent tools: approve_hitl_request is a human-controlled surface.")

    worked = model.capacity_for(100, 2.0)
    return output.model_copy(
        update={
            "artifact": {
                "law": "L = λW",
                "worked example": "100/hr × 2 min",
                "reviewer-hours per hour": worked.reviewer_hours_per_hour,
                "reviewers required": worked.reviewers_required,
                "stable at 3": model.is_stable(100, 2.0, 3),
                "queue has a drain": True,
            }
        }
    )


if __name__ == "__main__":
    main(run)
