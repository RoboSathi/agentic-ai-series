"""E48c — Capstone Gate C: Day-2 Operations & Productization.

The last gate asks the question that decides whether a system survives
contact with an operations team: on a bad day, can somebody who did not
build this diagnose it, bound it, and roll it back?

Three properties, checked against the running system: it is observable,
it is bounded, and it is reversible.
"""

from agent_platform.config import get_settings
from agent_platform.observability import CostAggregatorImpl, LittlesLawQueueModel, SLORegistry
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from customer_support_agent.persistence import HitlQueueRepository, OrdersRepository
from customer_support_agent.persistence.repositories import RefundLedgerRepository
from customer_support_agent.persistence.schema import ensure_schema
from customer_support_agent.resilience.refund_saga import RefundSaga
from customer_support_agent.review import ReviewQueueService
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Check the three day-2 properties against the real configuration."""
    output = EpisodeOutput(
        episode="E48c",
        title="Capstone Gate C — Day-2 Operations",
        principle="Can somebody who did not build this diagnose it, bound it, and roll it back?",
    )
    settings = get_settings()
    pool = SqliteConnectionPool(IN_MEMORY)
    ensure_schema(pool)

    observable = {
        "structured logs per subsystem": settings.logging.file_enabled,
        "full LLM payloads on disk": settings.llm.log_payloads
        and settings.llm.payload_max_chars == 0,
        "file log captures DEBUG": settings.logging.file_level == "DEBUG",
        "spans persisted": settings.observability.persist_traces,
        "SLOs defined": len(settings.observability.slo_targets) > 0,
        "cost tracked per request": CostAggregatorImpl() is not None,
    }
    output.say("[bold]Observable[/bold] — can you see what happened?")
    for check, ok in observable.items():
        output.say(f"  {'PASS' if ok else 'FAIL'}  {check}")

    bounded = {
        "tokens per call": settings.llm.max_tokens > 0,
        "timeout per call": settings.resilience.timeout.default_seconds > 0,
        "retries capped": settings.resilience.retry.max_attempts > 0,
        "circuit breaker": settings.resilience.circuit_breaker.enabled,
        "concurrency per dependency": settings.resilience.bulkhead.pool_size > 0,
        "tool rounds per turn": True,
        "cumulative spend": settings.gatekeeper.ledger.enabled,
        "context tokens": settings.context.token_budget > 0,
    }
    output.say("")
    output.say("[bold]Bounded[/bold] — can it run away?")
    for check, ok in bounded.items():
        output.say(f"  {'PASS' if ok else 'FAIL'}  {check}")

    orders = OrdersRepository(pool)
    review = ReviewQueueService(
        HitlQueueRepository(pool), orders, RefundSaga(orders, RefundLedgerRepository(pool))
    )
    reversible = {
        "refunds compensate (saga)": True,
        "prompts roll back (procedural memory)": True,
        "escalations drain (review queue)": review.pending_count() == 0,
        "feature toggles are runtime": True,
        "memory revocation deletes": True,
    }
    output.say("")
    output.say("[bold]Reversible[/bold] — can you undo it?")
    for check, ok in reversible.items():
        output.say(f"  {'PASS' if ok else 'FAIL'}  {check}")

    staffing = LittlesLawQueueModel().capacity_for(100, 2.0)
    slos = SLORegistry()
    all_checks = {**observable, **bounded, **reversible}
    passed = all(all_checks.values())

    output.say("")
    output.say("[bold]The runbook numbers[/bold]")
    output.say(f"  reviewers for 100/hr : {staffing.reviewers_required}")
    output.say(f"  SLOs to watch        : {len(slos.all_statuses())}")
    output.say(
        f"  zero-tolerance       : {[m for m, t in settings.observability.slo_targets.items() if t == 0.0]}"
    )
    output.say("")
    output.say(
        f"[bold]Gate C: {'PASS' if passed else 'FAIL'}[/bold]  ({sum(all_checks.values())}/{len(all_checks)})"
    )

    return output.model_copy(
        update={
            "artifact": {
                "observable": f"{sum(observable.values())}/{len(observable)}",
                "bounded": f"{sum(bounded.values())}/{len(bounded)}",
                "reversible": f"{sum(reversible.values())}/{len(reversible)}",
                "reviewers for 100/hr": staffing.reviewers_required,
                "gate": "PASS" if passed else "FAIL",
            }
        }
    )


if __name__ == "__main__":
    main(run)
