"""E35 — Resilience Is a Control Plane Too.

A refund is not one action. It is validate, reserve, issue, update,
notify — five local steps, each of which can fail after the ones before
it have already committed. That is a saga, and what makes it a saga
rather than a hopeful sequence is that every step has a defined
compensation.

The second half of the episode is the pairing the thesis rests on. The
customer asks for the same refund twice, which is exactly what a real
customer does when a reply is slow. Without an idempotency key that is
two refunds. With one it is the same refund, answered twice — and the
number at the end of the run is the whole argument.
"""

from agentic_commons.money import Money
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.domain.identity import current_customer
from customer_support_agent.persistence.repositories import RefundLedgerRepository
from episodes._shared import EpisodeRun


async def flow(run: EpisodeRun) -> None:
    """Verify, refund, ask for the same refund again, then count the money."""
    verify, *requests = run.settings.turns

    # Measured after verification, because "refunded to this customer" is
    # not a question that can be asked before there is a customer.
    await run.ask(verify)
    before = _refunded()

    states = [await run.ask(turn) for turn in requests]

    run.note("")
    for turn, state in zip(requests, states):
        run.note(f"{turn[:52]:<54} tool calls: {len(state.tool_results)}")

    run.note("")
    run.note(f"Refunded to this customer before: {before}")
    run.note(f"Refunded to this customer after : {_refunded()}")
    run.note("A retried request is the same request. That is what idempotency buys.")


def _refunded() -> Money:
    """What this customer has actually been refunded, reversals excluded."""
    customer = current_customer()
    if customer is None:
        return Money.zero()
    return RefundLedgerRepository(SqliteConnectionPool()).total_for_user(customer.customer_id)
