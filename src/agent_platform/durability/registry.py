"""The run registry: how each thread ended, in one queryable table.

The checkpointer is the source of a thread's *state*; it cannot answer
"which runs need attention?" without deserializing every checkpoint in
the file. This table answers exactly that, cheaply and synchronously, so
start-up recovery and the operator console never need an event loop at all.
"""

from datetime import datetime

from agent_platform.core.persistence import PlatformStore
from agent_platform.durability.types import RESUMABLE_STATUSES, RunRecord, RunStatus
from agent_platform.observability.types import utc_now

_TABLE = "platform_runs"

# Recorded as the error of a run whose process exited mid-run.
ORPHANED_REASON = "process exited before the run finished"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    thread_id     TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    subject_id    TEXT NOT NULL,
    status        TEXT NOT NULL,
    preview       TEXT NOT NULL,
    next_node     TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    error         TEXT NOT NULL,
    attempts      INTEGER NOT NULL,
    waiting_on    TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    trace_id      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_{_TABLE}_status ON {_TABLE} (agent_id, status);
CREATE INDEX IF NOT EXISTS idx_{_TABLE}_session ON {_TABLE} (agent_id, session_id, updated_at);
"""

_COLUMNS = (
    "thread_id, agent_id, session_id, subject_id, status, preview, next_node, "
    "checkpoint_id, error, attempts, waiting_on, started_at, updated_at, trace_id"
)
# Columns added after the table first shipped, with the definition that adds them.
_ADDED_COLUMNS = {"trace_id": "TEXT NOT NULL DEFAULT ''"}


class RunRegistry:
    """Durable index of runs, scoped to one agent."""

    def __init__(self, store: PlatformStore) -> None:
        self._store = store
        self._agent_id = store.agent_id
        self._store.execute_script(_SCHEMA)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Bring a table created by an earlier release up to the current columns."""
        present = {
            str(row["name"]) for row in self._store.iter_rows(f"PRAGMA table_info({_TABLE})")
        }
        for column, definition in _ADDED_COLUMNS.items():
            if column not in present:
                self._store.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {column} {definition}")

    def start(
        self, thread_id: str, session_id: str, subject_id: str, preview: str, trace_id: str = ""
    ) -> RunRecord:
        """Register a new run as running, with the trace its log lines carry."""
        record = RunRecord(
            thread_id=thread_id,
            agent_id=self._agent_id,
            session_id=session_id,
            subject_id=subject_id,
            preview=preview,
            attempts=1,
            trace_id=trace_id,
        )
        self._store.execute(
            f"INSERT OR REPLACE INTO {_TABLE} ({_COLUMNS}) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _row(record),
        )
        return record

    def mark(
        self,
        thread_id: str,
        status: RunStatus,
        *,
        next_node: str = "",
        checkpoint_id: str = "",
        error: str = "",
        waiting_on: str = "",
    ) -> None:
        """Record how a run stopped (or that it is running again)."""
        self._store.execute(
            f"UPDATE {_TABLE} SET status = ?, next_node = ?, checkpoint_id = ?, error = ?, "
            "waiting_on = ?, updated_at = ? WHERE agent_id = ? AND thread_id = ?",
            (
                status,
                next_node,
                checkpoint_id,
                error,
                waiting_on,
                utc_now().isoformat(),
                self._agent_id,
                thread_id,
            ),
        )

    def attempt(self, thread_id: str) -> None:
        """Count one more attempt at a run (a resume)."""
        self._store.execute(
            f"UPDATE {_TABLE} SET attempts = attempts + 1, status = 'running', updated_at = ? "
            "WHERE agent_id = ? AND thread_id = ?",
            (utc_now().isoformat(), self._agent_id, thread_id),
        )

    def get(self, thread_id: str) -> RunRecord | None:
        row = self._store.query_one(
            f"SELECT {_COLUMNS} FROM {_TABLE} WHERE agent_id = ? AND thread_id = ?",
            (self._agent_id, thread_id),
        )
        return _record(row) if row else None

    def find(self, prefix: str) -> RunRecord | None:
        """A run by thread id or unique prefix — what an operator types."""
        rows = list(
            self._store.iter_rows(
                f"SELECT {_COLUMNS} FROM {_TABLE} WHERE agent_id = ? AND thread_id LIKE ? "
                "ORDER BY updated_at DESC",
                (self._agent_id, f"{prefix}%"),
            )
        )
        return _record(rows[0]) if len(rows) == 1 else None

    def waiting_on(self, key: str) -> RunRecord | None:
        """The interrupted run waiting on one external decision, if any."""
        row = self._store.query_one(
            f"SELECT {_COLUMNS} FROM {_TABLE} WHERE agent_id = ? AND status = 'interrupted' "
            "AND waiting_on = ? ORDER BY updated_at DESC",
            (self._agent_id, key),
        )
        return _record(row) if row else None

    def recent(self, limit: int, statuses: tuple[RunStatus, ...] = ()) -> list[RunRecord]:
        """Most recently updated runs, optionally only some statuses."""
        clause, params = _status_clause(statuses)
        rows = self._store.iter_rows(
            f"SELECT {_COLUMNS} FROM {_TABLE} WHERE agent_id = ?{clause} "
            "ORDER BY updated_at DESC LIMIT ?",
            (self._agent_id, *params, limit),
        )
        return [_record(row) for row in rows]

    def resumable(self, limit: int) -> list[RunRecord]:
        """Runs an operator can resume: failed part-way, or waiting on a human."""
        return self.recent(limit, RESUMABLE_STATUSES)

    def orphaned(self) -> list[RunRecord]:
        """Runs still marked running — the process died under them."""
        return self.recent(-1, ("running",))

    def recover_orphans(self) -> list[RunRecord]:
        """Mark runs a dead process left 'running' as resumable.

        Called once at process start-up, never mid-flight: a run marked
        running at start-up has no process left executing it, and its
        last checkpoint is exactly where a resume should pick it up.
        """
        orphans = self.orphaned()
        for record in orphans:
            self.mark(
                record.thread_id,
                "failed_resumable",
                next_node=record.next_node,
                checkpoint_id=record.checkpoint_id,
                error=ORPHANED_REASON,
            )
        return orphans

    def latest_in_session(self, session_id: str) -> RunRecord | None:
        """A session's most recent run, however it ended — what a console offers to resume."""
        row = self._store.query_one(
            f"SELECT {_COLUMNS} FROM {_TABLE} WHERE agent_id = ? AND session_id = ? "
            "ORDER BY updated_at DESC",
            (self._agent_id, session_id),
        )
        return _record(row) if row else None

    def latest_completed(self, session_id: str) -> RunRecord | None:
        """The last finished run of a session — where its conversation stands."""
        row = self._store.query_one(
            f"SELECT {_COLUMNS} FROM {_TABLE} WHERE agent_id = ? AND session_id = ? "
            "AND status IN ('completed', 'interrupted') ORDER BY updated_at DESC",
            (self._agent_id, session_id),
        )
        return _record(row) if row else None

    def expired(self, before: datetime) -> list[str]:
        """Finished threads last touched before a cut-off (retention)."""
        rows = self._store.iter_rows(
            f"SELECT thread_id FROM {_TABLE} WHERE agent_id = ? AND status = 'completed' "
            "AND updated_at < ?",
            (self._agent_id, before.isoformat()),
        )
        return [str(row["thread_id"]) for row in rows]

    def forget(self, thread_id: str) -> None:
        self._store.execute(
            f"DELETE FROM {_TABLE} WHERE agent_id = ? AND thread_id = ?",
            (self._agent_id, thread_id),
        )


def _status_clause(statuses: tuple[RunStatus, ...]) -> tuple[str, tuple[str, ...]]:
    if not statuses:
        return "", ()
    marks = ", ".join("?" for _ in statuses)
    return f" AND status IN ({marks})", tuple(statuses)


def _row(record: RunRecord) -> tuple[object, ...]:
    return (
        record.thread_id,
        record.agent_id,
        record.session_id,
        record.subject_id,
        record.status,
        record.preview,
        record.next_node,
        record.checkpoint_id,
        record.error,
        record.attempts,
        record.waiting_on,
        record.started_at.isoformat(),
        record.updated_at.isoformat(),
        record.trace_id,
    )


def _record(row: dict[str, object]) -> RunRecord:
    return RunRecord.model_validate(row)
