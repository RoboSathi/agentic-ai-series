"""Run one query through the agent: `python -m customer_support_agent`.

The developer entry point `make run` invokes. It used to name
`customer_support_agent.dev.run_demo`, a module that does not exist and
has not for some time — the target failed for anyone who typed it.

Deliberately small. It composes nothing of its own: the agent comes from
the same `build_agent` the TUI and every episode use, so what this prints
is what the product does rather than a demo-shaped imitation of it.
"""

import asyncio
import sys

from agentic_commons.logging import get_logger
from customer_support_agent.config import get_agent_settings
from customer_support_agent.graph import build_agent

# What to ask when nobody says. The verification code comes from the same
# config the seeded database was built from, and the order is one that
# customer owns and that is still in transit — so a fresh checkout shows
# the tracking answer rather than a refusal.
_DEFAULT_QUERY = "where is my order 10510?"


async def _run(query: str) -> int:
    """Answer one query, printing the reply and the decisions behind it."""
    settings = get_agent_settings()
    agent = build_agent()
    logger = get_logger("customer_support_agent")

    # Authenticate first: every order action is scoped to a verified
    # customer, so a demo that skipped this would exercise a path the
    # product does not have.
    await agent.handle(settings.demo_user_code, session_id="demo")
    state = await agent.handle(query, session_id="demo")

    print(f"\n> {query}\n")
    print(state.response or "(no response)")
    if state.decisions:
        print("\nDecisions:")
        for decision in state.decisions:
            print(f"  - {decision}")
    logger.info("Demo turn complete", query=query, escalated=state.escalated)
    return 0


def main() -> int:
    """Entry point: the query from the command line, or the default."""
    query = " ".join(sys.argv[1:]).strip() or _DEFAULT_QUERY
    return asyncio.run(_run(query))


if __name__ == "__main__":
    raise SystemExit(main())
