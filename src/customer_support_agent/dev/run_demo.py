"""Demo entry point: run one sample query through the customer support agent.

Requires the database to have been seeded (`make seed-db`) so there's a
real order for the demo query to find.
"""

from agentic_commons.logging import get_logger
from customer_support_agent.core.graph import run_agent

_DEMO_QUERY = "What is the status of order 12345, and can I get a refund?"


def main() -> str:
    """Run one demo query through the agent and print the result."""
    logger = get_logger("customer_support_agent")
    logger.info(f"[bold bright_green]Query:[/bold bright_green] {_DEMO_QUERY}")

    response = run_agent(_DEMO_QUERY)

    logger.info(f"[bold bright_green]Response:[/bold bright_green] {response}")
    return response


if __name__ == "__main__":
    main()
