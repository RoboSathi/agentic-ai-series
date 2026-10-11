"""Dream events: what one completed piece of agent work leaves for the next dream.

A turn, or for an agent with longer work a whole run, ends by writing one
event to the dream inbox: what was asked, what was answered, what the
agent did and saw — and any **signals**, typed findings the agent names
in its own vocabulary (`refused`, `criterion_failed`, `source_rejected`).

An event makes no long-term-memory decision. Whether any of it becomes
an episode, a fact or a skill is the dream's call, made later, over many
events at once — which is the point: one turn cannot tell a coincidence
from a pattern.
"""

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryObservation(BaseModel):
    """One observed piece of information and where it came from."""

    content: Any
    source: str = ""


class MemoryAction(BaseModel):
    """One action the agent actually attempted."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    source: str = ""


class DreamSignal(BaseModel):
    """A typed finding the agent recorded, in its own words.

    `kind` and `value` together are what recurs (`refused` / "order is
    already shipped"); `scope` is the situation it recurred in, written as
    a skill trigger (`intent:<name>`, `kind:<name>`), so a
    skill learned from it applies where the finding was made.
    """

    kind: str
    value: str
    scope: str = ""


class DreamEvent(BaseModel):
    """One completed turn or run, as the dream inbox holds it."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_id: str
    session_id: str
    # Whose work it was; empty for an agent's own (a research run).
    subject_id: str = ""
    # `turn` by default (`memory.dream.turn_event_kind`); an agent may record others.
    kind: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_input: str = ""
    assistant_output: str = ""
    observations: list[MemoryObservation] = Field(default_factory=list)
    actions: list[MemoryAction] = Field(default_factory=list)
    signals: list[DreamSignal] = Field(default_factory=list)
    # The situations this work happened in, as skill triggers (see DreamSignal.scope).
    scope: list[str] = Field(default_factory=list)
    # Structured detail an agent's own consolidators read (a run's verdicts, say).
    outcome: dict[str, Any] = Field(default_factory=dict)
    # Set when a dream has read it; None while it waits.
    consolidated_at: datetime | None = None


@runtime_checkable
class DreamEventSink(Protocol):
    """Destination for completed-work events."""

    def emit(self, event: DreamEvent) -> None: ...


class InboxCounts(BaseModel):
    """How much of the inbox a dream has worked through."""

    unconsolidated: int = 0
    consolidated: int = 0
