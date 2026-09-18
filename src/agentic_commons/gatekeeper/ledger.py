"""The cumulative ledger (E02, LLD §11).

Every individual action can be within policy while the sequence is not.
Eighty separate $50 refunds are eighty permitted actions and one $4,000
loss — the structuring attack, borrowed intact from financial crime,
where a reportable transaction is split below the reporting threshold.

A per-call check cannot see this by construction: it is given one call
and asked whether that call is allowed. The ledger is the component that
holds the memory a per-call check lacks, which is why it is a first-class
part of the mediation point rather than an analytics job that notices
afterwards.
"""

from datetime import timedelta

from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.types import LedgerEntry, LedgerState
from agentic_commons.logging import get_logger, logged
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gatekeeper_ledger (
    entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ledger_key   TEXT NOT NULL,
    action_class TEXT NOT NULL,
    actor        TEXT NOT NULL,
    amount       REAL NOT NULL,
    trace_id     TEXT NOT NULL,
    recorded_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_window
    ON gatekeeper_ledger (ledger_key, action_class, recorded_at);
"""


class CumulativeLedger:
    """Running totals per (ledger key, action class) inside a rolling window."""

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        settings = get_settings().gatekeeper.ledger
        self._pool = pool or SqliteConnectionPool()
        self._enabled = settings.enabled
        self._window = timedelta(hours=settings.window_hours)
        self._action_classes = settings.action_classes
        self._thresholds = settings.thresholds_usd
        self._logger = get_logger("gatekeeper")
        self._pool.execute_script(_SCHEMA)

    @property
    def enabled(self) -> bool:
        """Whether cumulative accounting is active."""
        return self._enabled

    def class_of(self, action: str) -> str:
        """Which cumulative budget an action consumes from.

        Mapped explicitly rather than derived from the action's name: an
        action whose class is inferred from its spelling silently stops
        being tracked the moment someone renames the tool.
        """
        bare = action.rsplit(".", maxsplit=1)[-1]
        return self._action_classes.get(bare, bare)

    def threshold_for(self, action_class: str) -> float:
        """The cumulative cap for an action class.

        An action class with no configured threshold is uncapped, and
        that is logged: an uncapped consequential action is a policy gap
        someone should see, not a silent default.
        """
        threshold = self._thresholds.get(action_class)
        if threshold is None:
            self._logger.debug("No cumulative threshold configured", action_class=action_class)
            return float("inf")
        return threshold

    def window_total(self, ledger_key: str, action_class: str) -> float:
        """Total consumed for a key and class inside the rolling window."""
        row = self._pool.query_one(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM gatekeeper_ledger "
            "WHERE ledger_key = ? AND action_class = ? AND recorded_at >= ?",
            (ledger_key, action_class, self._window_start()),
        )
        return float(row["total"]) if row else 0.0

    def state(self, ledger_key: str, action_class: str, pending: float = 0.0) -> LedgerState:
        """Where this key stands, optionally including a call not yet made.

        `pending` is what makes this a gate rather than a report: the
        eighty-first refund is refused *before* it is issued, because the
        check asks "would this take us over?" and not "did it?".
        """
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
        self._pool.execute(
            "INSERT INTO gatekeeper_ledger (ledger_key, action_class, actor, amount, "
            "trace_id, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
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
        row = self._pool.query_one(
            "SELECT COUNT(*) AS n FROM gatekeeper_ledger "
            "WHERE ledger_key = ? AND action_class = ? AND recorded_at >= ?",
            (ledger_key, action_class, self._window_start()),
        )
        return int(row["n"]) if row else 0

    def entries(self, ledger_key: str, action_class: str) -> list[LedgerEntry]:
        """Every entry in the window — the evidence behind a refusal."""
        return [
            LedgerEntry.model_validate({k: v for k, v in row.items() if k != "entry_id"})
            for row in self._pool.iter_rows(
                "SELECT * FROM gatekeeper_ledger WHERE ledger_key = ? AND action_class = ? "
                "AND recorded_at >= ? ORDER BY entry_id",
                (ledger_key, action_class, self._window_start()),
            )
        ]

    def _window_start(self) -> str:
        """ISO timestamp marking the start of the rolling window."""
        return (utc_now() - self._window).isoformat()
