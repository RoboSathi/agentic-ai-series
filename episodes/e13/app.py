"""E13 — Memory Architecture: Working, Semantic, Episodic, Procedural.

Four stores, and the claim that the difference between them is semantic
rather than technical. A preference ("never call me") is generally true
about this customer and belongs in semantic memory. An order lookup is
something that happened and belongs in episodic. The order currently
under discussion is active right now and belongs in working memory —
which is the store the fourth turn depends on, because "was that one
delivered on time?" names nothing at all.

The conversation is the demonstration. What it proves is in the last
turn: the agent answers a question whose subject exists only in the
previous turn, and recalls a preference stated three turns earlier.
"""

from episodes._shared import EpisodeRun


async def flow(run: EpisodeRun) -> None:
    """Hold the conversation, then show what each store ended up holding."""
    await run.converse()

    memory = run.agent.memory
    if memory is None:
        run.note("Memory is disabled in this episode's config — nothing to show.")
        return

    active = memory.active_window(run.session_id)
    run.note("")
    run.note(f"Working memory — active this conversation: {len(active)} facts")
    for fact in active:
        run.note(f"  · {fact.key}: {fact.value}")
