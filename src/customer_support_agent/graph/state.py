"""What flows between graph nodes.

A pydantic model rather than a loose dict, so a node returning the wrong
shape fails where it happened rather than three nodes later, where the
value is finally read.

It lives with the graph rather than with the domain because it is not a
domain concept: an `Order` is something this business has, while
`AgentState` is an artefact of running a turn through a state machine.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from agentic_commons.llm.types import Message


def _merge_messages(existing: list[Message], incoming: list[Message]) -> list[Message]:
    """Reducer: graph nodes append to the transcript rather than replace it."""
    return [*existing, *incoming]


class AgentState(BaseModel):
    """What flows between graph nodes.

    A pydantic model rather than a loose dict so that a node returning
    the wrong shape fails where it happened, not three nodes later where
    the value is finally read.
    """

    query: str = ""
    session_id: str = ""
    actor: str = ""
    customer_id: int | None = None
    messages: Annotated[list[Message], _merge_messages] = Field(default_factory=list)
    context: str = ""
    tool_rounds: int = 0
    tool_results: list[str] = Field(default_factory=list)
    # Only successfully completed tool results may ground customer-facing
    # business claims. ``tool_results`` intentionally retains blocked/failed
    # observations for the model and audit trail; this field is the
    # authority boundary used by reflection and hallucination protection.
    authoritative_tool_results: list[str] = Field(default_factory=list)
    authoritative_facts: dict[str, Any] = Field(default_factory=dict)
    decisions: list[str] = Field(default_factory=list)
    escalated: bool = False
    response: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
