"""Developer entry point — not shipped logic (LLD §21).

Runs a handful of refund conversations against the live model so the
whole lifecycle can be watched from a terminal. `make run` calls this.
"""

import asyncio

from agentic_commons.logging import get_logger
from customer_support_agent.core import build_agent

_QUERIES = [
    "What's the status of order 10042?",
    "Order 10042 arrived damaged, I'd like a refund.",
    "I want a refund for order 10007, it cost me over two hundred dollars.",
]


async def main() -> None:
    """Run the demo conversations end to end."""
    logger = get_logger("customer_support_agent")
    agent = build_agent()
    for index, query in enumerate(_QUERIES, start=1):
        logger.info(f"[bold]Query {index}[/bold]: {query}")
        state = await agent.handle(query, session_id=f"demo-{index}")
        for decision in state.decisions:
            logger.info(f"  · {decision}")
        logger.info(f"[bright_green]Response[/bright_green]: {state.response}")


if __name__ == "__main__":
    asyncio.run(main())
