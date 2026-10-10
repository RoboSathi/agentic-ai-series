"""`python -m customer_support_agent.evals.intent`: run the bake-off, print the report's path."""

import asyncio

from customer_support_agent.evals.intent.bakeoff import run

if __name__ == "__main__":
    print(asyncio.run(run()))
