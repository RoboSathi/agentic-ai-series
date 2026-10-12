"""The review queue: one table, in the agent's own database, for every kind.

Before, a refund waited in one table, a quarantined message in a second
and a doubtful memory in a third, each with its own command and its own
idea of a decision. Now each is a row here with a `kind`; the stores that
used to own a table write through this one, so nothing a person must
decide can be missed by looking in the wrong place.
"""

import json
from collections.abc import Callable
from typing import Any

from agent_platform.config import get_settings
from agent_platform.persistence import PlatformStore, SqliteConnectionPool, as_platform_store
from agent_platform.primitives.clock import utc_now
from agent_platform.review.types import ReviewItem, ReviewStatus

_PENDING = "pending"

Redactor = Callable[[str], str]


def _redacted(value: Any, redact: Redactor) -> Any:
    """`value` with every string in it, however deep, passed through `redact`."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redacted(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(item, redact) for item in value]
    return value


class ReviewQueue:
    """Durable review items for one agent."""

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
        redact: Redactor | None = None,
    ) -> None:
        # What is written is what a person later reads and an audit later quotes, so
        # personal data is taken out on the way in, not on the way to a screen.
        self._redact = redact
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._table = get_settings().review.table
        self._store.execute_script(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id    TEXT NOT NULL,
                kind        TEXT NOT NULL,
                status      TEXT NOT NULL,
                session_id  TEXT NOT NULL DEFAULT '',
                subject_id  TEXT NOT NULL DEFAULT '',
                summary     TEXT NOT NULL DEFAULT '',
                detail      TEXT NOT NULL DEFAULT '{{}}',
                created_at  TEXT NOT NULL,
                decided_by  TEXT,
                decided_at  TEXT,
                comment     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_{self._table}_pending
                ON {self._table} (agent_id, status, kind);
            """
        )

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def add(self, item: ReviewItem) -> ReviewItem:
        """Queue an item; returns it with its id."""
        if self._redact is not None:
            item = item.model_copy(
                update={
                    "summary": self._redact(item.summary),
                    "detail": _redacted(item.detail, self._redact),
                }
            )
        self._store.execute(
            f"INSERT INTO {self._table} (agent_id, kind, status, session_id, subject_id, "
            "summary, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._agent_id,
                item.kind,
                item.status,
                item.session_id,
                item.subject_id,
                item.summary,
                json.dumps(item.detail, default=str),
                item.created_at.isoformat(),
            ),
        )
        return item.model_copy(update={"id": self.last_id(), "agent_id": self._agent_id})

    def get(self, item_id: int) -> ReviewItem | None:
        row = self._store.query_one(
            f"SELECT * FROM {self._table} WHERE agent_id = ? AND id = ?", (self._agent_id, item_id)
        )
        return _item(row) if row else None

    def items(
        self,
        *,
        status: ReviewStatus | None = "pending",
        kinds: tuple[str, ...] = (),
        subject: str | None = None,
        limit: int | None = None,
    ) -> list[ReviewItem]:
        """Items oldest first; narrowed by status, kinds and subject (None: any subject)."""
        clauses, params = ["agent_id = ?"], [self._agent_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if kinds:
            clauses.append(f"kind IN ({', '.join('?' for _ in kinds)})")
            params += list(kinds)
        if subject is not None:
            clauses.append("subject_id = ?")
            params.append(subject)
        sql = f"SELECT * FROM {self._table} WHERE {' AND '.join(clauses)} ORDER BY id"
        rows = [_item(row) for row in self._store.iter_rows(sql, tuple(params))]
        return rows if limit is None else rows[-limit:]

    def decide(self, item_id: int, status: ReviewStatus) -> bool:
        """Settle a pending item's status; False when it was not pending."""
        result = self._store.execute(
            f"UPDATE {self._table} SET status = ?, decided_at = ? "
            "WHERE agent_id = ? AND id = ? AND status = ?",
            (status, utc_now().isoformat(), self._agent_id, item_id, _PENDING),
        )
        return result.affected > 0

    def sign(self, item_id: int, decided_by: str, comment: str) -> None:
        """Record who decided and why, whichever path settled the status."""
        if self._redact is not None:
            comment = self._redact(comment)
        self._store.execute(
            f"UPDATE {self._table} SET decided_by = ?, comment = ?, "
            "decided_at = COALESCE(decided_at, ?) WHERE agent_id = ? AND id = ?",
            (decided_by, comment, utc_now().isoformat(), self._agent_id, item_id),
        )

    def remove(self, item_id: int) -> None:
        """Delete one item — only as the compensation for a failed enqueue."""
        self._store.execute(
            f"DELETE FROM {self._table} WHERE agent_id = ? AND id = ?", (self._agent_id, item_id)
        )

    def counts(self, kinds: tuple[str, ...] = ()) -> dict[str, int]:
        """How many items are in each status (of these kinds, if named)."""
        where, params = (
            (f"AND kind IN ({', '.join('?' for _ in kinds)})", kinds) if kinds else ("", ())
        )
        return {
            str(row["status"]): int(row["n"])
            for row in self._store.iter_rows(
                f"SELECT status, COUNT(*) AS n FROM {self._table} WHERE agent_id = ? {where} "
                "GROUP BY status",
                (self._agent_id, *params),
            )
        }

    def last_id(self) -> int:
        row = self._store.query_one(
            f"SELECT MAX(id) AS id FROM {self._table} WHERE agent_id = ?", (self._agent_id,)
        )
        return int(row["id"]) if row and row["id"] is not None else 0

    def purge(self, kinds: tuple[str, ...]) -> int:
        """Remove every item of these kinds. Operator reset only (`make clean-memory`)."""
        result = self._store.execute(
            f"DELETE FROM {self._table} WHERE agent_id = ? "
            f"AND kind IN ({', '.join('?' for _ in kinds)})",
            (self._agent_id, *kinds),
        )
        return result.affected


def _item(row: dict[str, Any]) -> ReviewItem:
    return ReviewItem.model_validate({**row, "detail": json.loads(row.get("detail") or "{}")})
