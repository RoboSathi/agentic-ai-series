"""Developer entry point — not shipped logic (LLD §21).

Walks one conversation through the whole flow: an off-topic question that
is refused without a model call, the verification handshake, then order
and refund questions against the verified customer's own orders.

`make run` calls this. It is one conversation rather than several so the
live model is called as few times as the demo allows.
"""

import asyncio

from agentic_commons.logging import get_logger
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.config import get_agent_settings
from customer_support_agent.core import OrdersRepository, build_agent
from customer_support_agent.core.repositories import UsersRepository


def _script(code: str) -> list[str]:
    """The demo conversation, built against the demo customer's own orders.

    Order ids are looked up rather than hard-coded: lookups are
    ownership-scoped, so an id belonging to somebody else would show
    "no order found" and read as a bug when the control is working.
    """
    pool = SqliteConnectionPool()
    user = UsersRepository(pool).get_by_code(code)
    script = [
        # Refused before any model call — this is not a general assistant.
        "What's the weather like today in London?",
        # Refused too: it needs to know who is asking before it looks up.
        "what is the status of my order?",
        # The handshake. Verified by a database lookup, not by a model.
        f"my code is {code}",
    ]
    if user is None:
        return script
    orders = OrdersRepository(pool).for_user(user.user_id)
    delivered = next((o for o in orders if o.order_status == "delivered"), None)
    in_flight = next((o for o in orders if o.order_status in {"pending", "shipped"}), None)
    if delivered:
        script += [
            f"what is the status of order {delivered.order_id}?",
            f"order {delivered.order_id} arrived damaged, I'd like a refund",
        ]
    if in_flight:
        script.append(f"I want a refund for order {in_flight.order_id}")
    return script


async def main() -> None:
    """Run the demo conversation end to end."""
    logger = get_logger("customer_support_agent")
    agent = build_agent()
    for query in _script(get_agent_settings().demo_user_code):
        logger.info(f"[bold]> {query}[/bold]")
        state = await agent.handle(query, session_id="demo")
        for decision in state.decisions:
            logger.info(f"  · {decision}")
        logger.info(f"[bright_green]{state.response}[/bright_green]")


if __name__ == "__main__":
    asyncio.run(main())
