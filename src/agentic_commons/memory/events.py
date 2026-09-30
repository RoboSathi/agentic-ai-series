"""Minimal event contract for asynchronous long-term-memory processing."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_commons.persistence import PlatformStore

# When a dream processed an event; NULL until one has.
_CONSOLIDATED = "consolidated_at"


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
    # Whose turn it was. Empty on events recorded before it was captured;
    # a dream then reads the subject from the session's episodic log.
    subject_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_input: str
    assistant_output: str
    observations: list[MemoryObservation] = Field(default_factory=list)
    actions: list[MemoryAction] = Field(default_factory=list)


@runtime_checkable
class MemoryEventSink(Protocol):
    """Destination for completed-turn memory events."""

    def emit(self, event: MemoryEvent) -> None: ...


class InboxCounts(BaseModel):
    """How much of the inbox a dream has worked through."""

    unconsolidated: int = 0
    consolidated: int = 0


class DreamInbox:
    """Small durable inbox for completed-turn memory events.

    A dream (`memory.consolidation`) reads what is unconsolidated, marks
    what it processed, and prunes what it processed long ago — so the
    inbox holds recent transcripts for audit and re-runs, not forever.
    """

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
        self._migrate()

    def _migrate(self) -> None:
        """Add `consolidated_at` to an inbox created before dreams existed."""
        columns = {
            str(row["name"]) for row in self._store.iter_rows(f"PRAGMA table_info({self._TABLE})")
        }
        if _CONSOLIDATED not in columns:
            self._store.execute(f"ALTER TABLE {self._TABLE} ADD COLUMN {_CONSOLIDATED} TEXT")

    def counts(self) -> InboxCounts:
        """Unconsolidated and consolidated events for this agent."""
        row = self._store.query_one(
            f"SELECT SUM({_CONSOLIDATED} IS NULL) AS pending, "
            f"SUM({_CONSOLIDATED} IS NOT NULL) AS done FROM {self._TABLE} WHERE agent_id = ?",
            (self._store.agent_id,),
        )
        return InboxCounts(
            unconsolidated=int((row or {}).get("pending") or 0),
            consolidated=int((row or {}).get("done") or 0),
        )

    def unconsolidated(self) -> dict[str, list[MemoryEvent]]:
        """Events no dream has processed, by session, oldest first within each."""
        sessions: dict[str, list[MemoryEvent]] = defaultdict(list)
        for row in self._store.iter_rows(
            f"SELECT payload FROM {self._TABLE} WHERE agent_id = ? AND {_CONSOLIDATED} IS NULL "
            "ORDER BY timestamp",
            (self._store.agent_id,),
        ):
            event = MemoryEvent.model_validate(json.loads(str(row["payload"])))
            sessions[event.session_id].append(event)
        return dict(sessions)

    def mark_consolidated(self, event_ids: list[str], at: datetime) -> None:
        """Record that a dream processed these events; a later dream skips them."""
        self._store.execute_many(
            f"UPDATE {self._TABLE} SET {_CONSOLIDATED} = ? WHERE agent_id = ? AND event_id = ?",
            [(at.isoformat(), self._store.agent_id, event_id) for event_id in event_ids],
        )

    def prune(self, consolidated_before: datetime) -> int:
        """Delete events consolidated before a cutoff; unprocessed ones are never pruned."""
        result = self._store.execute(
            f"DELETE FROM {self._TABLE} WHERE agent_id = ? AND {_CONSOLIDATED} IS NOT NULL "
            f"AND {_CONSOLIDATED} < ?",
            (self._store.agent_id, consolidated_before.isoformat()),
        )
        return result.affected

    def purge(self) -> int:
        """Remove every one of this agent's rows. Operator reset only.

        Deliberately outside the write pipeline: this is `make clean-memory`
        wiping a demo back to empty, not a memory operation with a
        principal — the same standing as rebuilding the database.
        """
        result = self._store.execute(
            f"DELETE FROM {self._TABLE} WHERE agent_id = ?", (self._store.agent_id,)
        )
        return result.affected

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
