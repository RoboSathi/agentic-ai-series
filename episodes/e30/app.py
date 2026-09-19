"""E30 — Prompt Injection Is a Control-Flow Problem.

The attack in this episode's script is not subtle, and it does not need
to be. It asks the agent, in the customer's own message, to grant itself
refund authority — and the interesting question is not whether the model
falls for it.

The thesis is that it does not matter. Injected text arrives as data, is
delimited as data, and the tool it is trying to reach is ruled on by the
Gatekeeper against the authority of whoever is actually asking. A
customer cannot authorize their own refund, so a customer *pretending to
be a system notice* cannot either. Detection is a useful layer. It is
not the one doing the work.
"""

from episodes._shared import EpisodeRun


async def flow(run: EpisodeRun) -> None:
    """Verify, look up an order, then attack the authority boundary."""
    states = await run.converse()

    run.note("")
    attacked = states[-1]
    run.note(f"Decisions on the injected turn: {len(attacked.decisions)}")
    for decision in attacked.decisions:
        run.note(f"  · {decision}")
    refunded = [result for result in attacked.tool_results if "process_refund" in str(result)]
    run.note("")
    run.note(
        f"Refunds executed: {len(refunded)} — the authority check does not depend "
        "on the model having noticed the attack."
    )
