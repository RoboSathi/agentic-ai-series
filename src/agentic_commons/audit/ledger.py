"""Append-only audit ledger (E33, LLD §5).

The append-only property is enforced by construction, not by convention:
this class exposes `record()` and `reconstruct()` and simply has no
method that issues UPDATE or DELETE. There is nothing to call, so there
is nothing to review for having called it.
"""

import json
from typing import Any, Protocol, runtime_checkable

from agentic_commons.audit.types import AuditEntry, Decision
from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.persistence import SqliteConnectionPool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_entries (
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id          TEXT    NOT NULL,
    actor             TEXT    NOT NULL,
    action            TEXT    NOT NULL,
    authority_context TEXT    NOT NULL,
    model_version     TEXT    NOT NULL,
    policy_version    TEXT    NOT NULL,
    data_version      TEXT    NOT NULL,
    decision          TEXT    NOT NULL,
    rationale         TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_entries (trace_id, seq);
"""

_INSERT = """
INSERT INTO audit_entries (
    trace_id, actor, action, authority_context, model_version,
    policy_version, data_version, decision, rationale, timestamp
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_TRACE = """
SELECT * FROM audit_entries WHERE trace_id = ? ORDER BY seq ASC
"""

_SELECT_DECISION = """
SELECT * FROM audit_entries WHERE decision = ? ORDER BY seq ASC
"""


@runtime_checkable
class AuditLedger(Protocol):
    """Append-only evidence chain over consequential decisions."""

    def record(self, entry: AuditEntry) -> None: ...

    def reconstruct(self, trace_id: str) -> list[AuditEntry]: ...


@instrument("audit")
class SqliteAuditLedger:
    """Audit entries in SQLite, ordered by insertion sequence.

    Causal order is the insert sequence rather than the wall-clock
    timestamp: two decisions inside the same millisecond must still
    reconstruct in the order they actually happened.
    """

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        settings = get_settings().audit
        self._pool = pool or SqliteConnectionPool()
        self._policy_version = settings.policy_version
        self._data_version = settings.data_version
        self._logger = get_logger("audit")
        self._pool.execute_script(_SCHEMA)

    @logged("audit")
    def record(self, entry: AuditEntry) -> None:
        """Append one entry. Versions default to the configured ones."""
        stamped = entry.model_copy(
            update={
                "policy_version": entry.policy_version or self._policy_version,
                "data_version": entry.data_version or self._data_version,
            }
        )
        self._pool.execute(
            _INSERT,
            (
                stamped.trace_id,
                stamped.actor,
                stamped.action,
                json.dumps(stamped.authority_context, default=str),
                stamped.model_version,
                stamped.policy_version,
                stamped.data_version,
                stamped.decision,
                stamped.rationale,
                stamped.timestamp.isoformat(),
            ),
        )

    def reconstruct(self, trace_id: str) -> list[AuditEntry]:
        """Every entry for one trace, in causal order."""
        return [self._to_entry(row) for row in self._pool.iter_rows(_SELECT_TRACE, (trace_id,))]

    def by_decision(self, decision: Decision) -> list[AuditEntry]:
        """Every entry with a given verdict — the escalation review feed."""
        return [self._to_entry(row) for row in self._pool.iter_rows(_SELECT_DECISION, (decision,))]

    def count(self) -> int:
        """Total entries recorded — used by the cross-subsystem invariants."""
        row = self._pool.query_one("SELECT COUNT(*) AS n FROM audit_entries")
        return int(row["n"]) if row else 0

    @staticmethod
    def _to_entry(row: dict[str, Any]) -> AuditEntry:
        """Rehydrate a stored row, JSON-decoding the authority context."""
        data = dict(row)
        data.pop("seq", None)
        data["authority_context"] = json.loads(data["authority_context"])
        return AuditEntry.model_validate(data)


class NullAuditLedger:
    """No-op ledger for episodes that predate the audit trail (E01–E32)."""

    def record(self, entry: AuditEntry) -> None:
        """Deliberately discards — an episode may not yet have an audit story."""

    def reconstruct(self, trace_id: str) -> list[AuditEntry]:
        """Always empty: nothing was ever recorded."""
        return []


def create_audit_ledger(pool: SqliteConnectionPool | None = None) -> AuditLedger:
    """Config-driven ledger factory (Factory pattern)."""
    if not get_settings().audit.enabled:
        return NullAuditLedger()
    return SqliteAuditLedger(pool)
