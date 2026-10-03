"""Pause a durable run until a decision arrives from outside it.

The generic half of every human-in-the-loop wait: interrupt the graph
with a payload that says what it is waiting on (so the decision can find
its way back to this exact thread through `RunRegistry.waiting_on`) and a
one-line summary for whoever is asked; return whatever decision the run
is resumed with.

LangGraph re-runs the whole node that called `interrupt()` on resume, so
call this from a node that does nothing else with side effects.
"""

from typing import Any

from langgraph.types import interrupt

from agent_platform.durability.runner import SUMMARY, WAITING_ON


def await_decision(waiting_on: str, summary: str, **fields: Any) -> Any:
    """Interrupt until resumed; return the resume value (the decision)."""
    return interrupt({WAITING_ON: waiting_on, SUMMARY: summary, **fields})
