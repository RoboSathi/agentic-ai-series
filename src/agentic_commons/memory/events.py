"""Minimal event contract for asynchronous long-term-memory processing."""

from datetime import datetime, timezone
import json
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_commons.persistence import PlatformStore


class MemoryObservation(BaseModel):
    """One observed piece of information and where it came from."""

    content: Any
    source: str = ""


class MemoryAction(BaseModel):
    """One action the agent actually attempted during the turn."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    source: str = ""


class MemoryEvent(BaseModel):
    """The raw outcome of one completed customer turn.

    It intentionally contains no long-term memory classification. The
    future dream/consolidation process decides whether this observation is
    episodic, semantic, procedural, or irrelevant.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_input: str
    assistant_output: str
    observations: list[MemoryObservation] = Field(default_factory=list)
    actions: list[MemoryAction] = Field(default_factory=list)


@runtime_checkable
class MemoryEventSink(Protocol):
    """Destination for completed-turn memory events."""

    def emit(self, event: MemoryEvent) -> None: ...


class DreamInbox:
    """Small durable inbox for completed-turn memory events."""

    _TABLE = "platform_dream_inbox"

    def __init__(self, store: PlatformStore) -> None:
        self._store = store
        self._store.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
                event_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )

    def emit(self, event: MemoryEvent) -> None:
        """Persist one event; repeated event ids are harmless."""
        self._store.execute(
            f"""
            INSERT OR IGNORE INTO {self._TABLE}
                (event_id, agent_id, session_id, timestamp, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.agent_id,
                event.session_id,
                event.timestamp.isoformat(),
                json.dumps(event.model_dump(mode="json")),
            ),
        )
