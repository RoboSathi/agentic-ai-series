"""Developer entry point — not shipped logic (LLD §21).

Runs a handful of refund conversations against the live model so the
whole lifecycle can be watched from a terminal. `make run` calls this.
"""

import asyncio

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.core import OrdersRepository, build_agent
from customer_support_agent.core.identity import Customer


def _queries(customer_id: str) -> list[str]:
    """Demo queries against orders this customer actually owns.

    Hard-coded order ids went stale the moment lookups became
    ownership-scoped: they belonged to somebody else, so the demo showed
    "no order found" and looked broken when the control was working.
    """
    orders = OrdersRepository(SqliteConnectionPool()).for_user(customer_id)
    delivered = next((o for o in orders if o.order_status == "delivered"), None)
    in_flight = next((o for o in orders if o.order_status in {"pending", "shipped"}), None)
    queries = ["Who are you and what can you help with?"]
    if delivered:
        queries += [
            f"What's the status of order {delivered.order_id}?",
            f"Order {delivered.order_id} arrived damaged, I'd like a refund.",
        ]
    if in_flight:
        queries.append(f"I want a refund for order {in_flight.order_id}.")
    return queries


async def main() -> None:
    """Run the demo conversations end to end."""
    logger = get_logger("customer_support_agent")
    agent = build_agent()
    customer = Customer(customer_id=get_settings().customer_support_agent.demo_customer_id)
    queries = _queries(customer.customer_id)
    for index, query in enumerate(queries, start=1):
        logger.info(f"[bold]Query {index}[/bold]: {query}")
        state = await agent.handle(query, session_id="demo", customer=customer)
        for decision in state.decisions:
            logger.info(f"  · {decision}")
        logger.info(f"[bright_green]Response[/bright_green]: {state.response}")


if __name__ == "__main__":
    asyncio.run(main())
