"""Cumulative action-spend ledger (LLD §11).

The ledger is platform-owned operational state. It is stored in the shared
SQLite file, but only through ``PlatformStore`` and always scoped by the
current ``agent_id``. That keeps the common Gatekeeper independent of any
agent's domain tables.
"""

from datetime import timedelta
from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.types import LedgerEntry, LedgerState
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_gatekeeper_ledger (
    entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id     TEXT NOT NULL,
    ledger_key   TEXT NOT NULL,
    action_class TEXT NOT NULL,
    actor        TEXT NOT NULL,
    amount       REAL NOT NULL,
    trace_id     TEXT NOT NULL,
    recorded_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_platform_gatekeeper_window
    ON platform_gatekeeper_ledger (agent_id, ledger_key, action_class, recorded_at);
"""


@instrument("gatekeeper")
class CumulativeLedger:
    """Running totals per (agent, ledger key, action class) inside a rolling window."""

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        settings = get_settings().gatekeeper.ledger
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._enabled = settings.enabled
        self._window = timedelta(hours=settings.window_hours)
        self._action_classes = settings.action_classes
        self._thresholds = settings.thresholds_usd
        self._logger = get_logger("gatekeeper")
        self._store.execute_script(_SCHEMA)
        self._migrate_legacy_table()

    @property
    def enabled(self) -> bool:
        """Whether cumulative accounting is active."""
        return self._enabled

    @property
    def agent_id(self) -> str:
        """Agent whose cumulative budget this ledger enforces."""
        return self._agent_id

    def class_of(self, action: str) -> str:
        """Which cumulative budget an action consumes from."""
        bare = action.rsplit(".", maxsplit=1)[-1]
        return self._action_classes.get(bare, bare)

    def is_budgeted(self, action: str) -> bool:
        """Whether this action is explicitly assigned to a cumulative budget.

        Cumulative accounting is an opt-in financial/risk control. An action
        that is not mapped is not silently assigned an infinite budget. This
        prevents read operations and non-financial writes such as queuing a
        review request from polluting the spend ledger.
        """
        bare = action.rsplit(".", maxsplit=1)[-1]
        return bare in self._action_classes

    def threshold_configured(self, action_class: str) -> bool:
        """Whether a finite policy threshold exists for a budget class."""
        return action_class in self._thresholds

    def threshold_for(self, action_class: str) -> float:
        """The cumulative cap for an action class."""
        threshold = self._thresholds.get(action_class)
        if threshold is None:
            self._logger.debug("No cumulative threshold configured", action_class=action_class)
            return float("inf")
        return threshold

    def window_total(self, ledger_key: str, action_class: str) -> float:
        """Total consumed for a key and class inside the rolling window."""
        row = self._store.query_one(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM platform_gatekeeper_ledger "
            "WHERE agent_id = ? AND ledger_key = ? AND action_class = ? AND recorded_at >= ?",
            (self._agent_id, ledger_key, action_class, self._window_start()),
        )
        return float(row["total"]) if row else 0.0

    def state(self, ledger_key: str, action_class: str, pending: float = 0.0) -> LedgerState:
        """Where this key stands, optionally including a call not yet made."""
        total = self.window_total(ledger_key, action_class) + pending
        threshold = self.threshold_for(action_class)
        return LedgerState(
            ledger_key=ledger_key,
            action_class=action_class,
            window_total=total,
            threshold=threshold,
            entry_count=self.entry_count(ledger_key, action_class),
            exceeded=total > threshold,
        )

    def would_exceed(self, ledger_key: str, action_class: str, amount: float) -> bool:
        """Whether adding this amount would cross the threshold."""
        if not self._enabled:
            return False
        return self.state(ledger_key, action_class, pending=amount).exceeded

    @logged("gatekeeper")
    def record(self, entry: LedgerEntry) -> LedgerState:
        """Record a consumed amount and return the resulting state."""
        if not self._enabled:
            return self.state(entry.ledger_key, entry.action_class)
        self._store.execute(
            "INSERT INTO platform_gatekeeper_ledger "
            "(agent_id, ledger_key, action_class, actor, amount, trace_id, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self._agent_id,
                entry.ledger_key,
                entry.action_class,
                entry.actor,
                entry.amount,
                entry.trace_id,
                entry.recorded_at.isoformat(),
            ),
        )
        return self.state(entry.ledger_key, entry.action_class)

    def record_and_check(self, entry: LedgerEntry) -> LedgerState:
        """Record, then report whether the threshold is now breached."""
        state = self.record(entry)
        if state.exceeded:
            self._logger.warning(
                "Cumulative threshold breached",
                ledger_key=entry.ledger_key,
                action_class=entry.action_class,
                total=state.window_total,
                threshold=state.threshold,
            )
        return state

    def entry_count(self, ledger_key: str, action_class: str) -> int:
        """How many entries a key has inside the window."""
        row = self._store.query_one(
            "SELECT COUNT(*) AS n FROM platform_gatekeeper_ledger "
            "WHERE agent_id = ? AND ledger_key = ? AND action_class = ? AND recorded_at >= ?",
            (self._agent_id, ledger_key, action_class, self._window_start()),
        )
        return int(row["n"]) if row else 0

    def entries(self, ledger_key: str, action_class: str) -> list[LedgerEntry]:
        """Every entry in the window — the evidence behind a refusal."""
        return [
            LedgerEntry.model_validate({k: v for k, v in row.items() if k != "entry_id"})
            for row in self._store.iter_rows(
                "SELECT agent_id, ledger_key, action_class, actor, amount, trace_id, recorded_at "
                "FROM platform_gatekeeper_ledger WHERE agent_id = ? AND ledger_key = ? "
                "AND action_class = ? AND recorded_at >= ? ORDER BY entry_id",
                (self._agent_id, ledger_key, action_class, self._window_start()),
            )
        ]

    def _window_start(self) -> str:
        """ISO timestamp marking the start of the rolling window."""
        return (utc_now() - self._window).isoformat()

    def _migrate_legacy_table(self) -> None:
        """Copy legacy gatekeeper rows into the agent-scoped platform table once."""
        if not self._table_exists("gatekeeper_ledger"):
            return
        columns = self._table_columns("gatekeeper_ledger")
        if "agent_id" in columns:
            rows = self._store.iter_rows(
                "SELECT agent_id, ledger_key, action_class, actor, amount, trace_id, recorded_at "
                "FROM gatekeeper_ledger WHERE agent_id = ?",
                (self._agent_id,),
            )
        else:
            rows = self._store.iter_rows(
                "SELECT ledger_key, action_class, actor, amount, trace_id, recorded_at "
                "FROM gatekeeper_ledger"
            )
        for row in rows:
            values = (
                row.get("agent_id", self._agent_id),
                row["ledger_key"],
                row["action_class"],
                row["actor"],
                row["amount"],
                row["trace_id"],
                row["recorded_at"],
            )
            exists = self._store.query_one(
                "SELECT 1 AS hit FROM platform_gatekeeper_ledger "
                "WHERE agent_id = ? AND ledger_key = ? AND action_class = ? "
                "AND actor = ? AND amount = ? AND trace_id = ? AND recorded_at = ? LIMIT 1",
                values,
            )
            if exists is None:
                self._store.execute(
                    "INSERT INTO platform_gatekeeper_ledger "
                    "(agent_id, ledger_key, action_class, actor, amount, trace_id, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    values,
                )

    def _table_exists(self, table: str) -> bool:
        row = self._store.query_one(
            "SELECT 1 AS hit FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        )
        return row is not None

    def _table_columns(self, table: str) -> set[str]:
        return {str(row["name"]) for row in self._store.iter_rows(f"PRAGMA table_info({table})")}
