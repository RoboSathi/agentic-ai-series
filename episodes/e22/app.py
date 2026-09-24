"""E22 — Tool Selection at Scale: Registry, Namespacing & Retrieval.

The registry knows every tool the agent has. The question this episode
answers is which of them a given turn should even be shown, and the
claim is that this is retrieval — scoring tool descriptions against the
query — rather than "list everything and let the model sort it out".

The demonstration is a contrast, so it runs the same turns twice: once
with retrieval-based selection on, once with it off. Nothing else about
the agent changes between the two runs. `set_enabled` flips the toggle
live, which is the same mechanism an operator has, and the agent is
rebuilt so the composition root re-reads it.
"""

from agentic_commons.config import is_enabled, set_enabled
from customer_support_agent.graph import build_agent
from episodes._shared import EpisodeRun


async def flow(run: EpisodeRun) -> None:
    """Run the conversation with selection on, then with it off."""
    run.note(f"Retrieval-based selection: {'on' if is_enabled('tools') else 'off'}")
    selected = await run.converse()

    run.note("")
    run.note("Same agent, same turns, selection off — the registry unfiltered.")
    set_enabled("tools", None, False)
    try:
        run.agent = build_agent()
        unfiltered = await run.converse()
    finally:
        set_enabled("tools", None, True)

    run.note("")
    for turn, with_it, without in zip(run.settings.turns, selected, unfiltered):
        run.note(f"{turn[:44]:<46} selected {_tools(with_it):>2} → offered {_tools(without):>2}")


def _tools(state: object) -> int:
    """How many tool calls that turn actually made."""
    return len(getattr(state, "tool_results", []))
