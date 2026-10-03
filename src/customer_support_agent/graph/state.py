"""What flows between graph nodes.

A pydantic model rather than a loose dict, so a node returning the wrong
shape fails where it happened rather than three nodes later, where the
value is finally read.

It lives with the graph rather than with the domain because it is not a
domain concept: an `Order` is something this business has, while
`AgentState` is an artefact of running a turn through a state machine.
"""

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.context import Context
from agent_platform.memory import WorkingMemory


class AgentState(BaseModel):
    """What flows between graph nodes.

    A pydantic model rather than a loose dict so that a node returning
    the wrong shape fails where it happened, not three nodes later where
    the value is finally read.
    """

    query: str = ""
    # The query with personal data removed — what the model is given.
    # Kept beside `query` rather than replacing it, because the two have
    # different jobs: deterministic in-process logic (ownership checks,
    # order resolution, mandates) must see what the customer actually
    # typed, while anything leaving this process must not. Empty until
    # the pii stage runs, and the model path falls back to `query` then.
    model_query: str = ""
    session_id: str = ""
    actor: str = ""
    customer_id: int | None = None
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)
    context: Context | None = None
    tool_rounds: int = 0
    decisions: list[str] = Field(default_factory=list)
    escalated: bool = False
    response: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
