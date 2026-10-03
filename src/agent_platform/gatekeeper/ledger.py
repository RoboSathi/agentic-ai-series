"""Cumulative action-spend ledger (LLD §11).

The ledger is platform-owned operational state. It is stored in the shared
SQLite file, but only through ``PlatformStore`` and always scoped by the
current ``agent_id``. That keeps the common Gatekeeper independent of any
agent's domain tables.
"""

from datetime import timedelta

from agent_platform.config import get_settings
from agent_platform.gatekeeper.types import LedgerEntry, LedgerState
from agent_platform.logging import get_logger, instrument, logged
from agent_platform.observability.types import utc_now
from agent_platform.persistence import PlatformStore, SqliteConnectionPool, as_platform_store
from agent_platform.primitives.money import Money

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_gatekeeper_ledger (
    entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id     TEXT NOT NULL,
    ledger_key   TEXT NOT NULL,
    action_class TEXT NOT NULL,
    actor        TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
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
        enabled: bool | None = None,
        action_classes: dict[str, str] | None = None,
        thresholds: dict[str, Money] | None = None,
        window_hours: int | None = None,
    ) -> None:
        defaults = get_settings().gatekeeper.ledger
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._enabled = defaults.enabled if enabled is None else enabled
        self._window = timedelta(
            hours=defaults.window_hours if window_hours is None else window_hours
        )
        self._action_classes = dict(
            defaults.action_classes if action_classes is None else action_classes
        )
        self._thresholds = (
            {name: Money.from_cents(cents) for name, cents in defaults.thresholds_cents.items()}
            if thresholds is None
            else dict(thresholds)
        )
        self._logger = get_logger("gatekeeper")
        self._store.execute_script(_SCHEMA)

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

    def threshold_for(self, action_class: str) -> Money | None:
        """The cumulative cap for an action class, or None when unbounded."""
        threshold = self._thresholds.get(action_class)
        if threshold is None:
            self._logger.debug("No cumulative threshold configured", action_class=action_class)
        return threshold

    def window_total(self, ledger_key: str, action_class: str) -> Money:
        """Total consumed for a key and class inside the rolling window — summed in cents."""
        row = self._store.query_one(
            "SELECT COALESCE(SUM(amount_cents), 0) AS total FROM platform_gatekeeper_ledger "
            "WHERE agent_id = ? AND ledger_key = ? AND action_class = ? AND recorded_at >= ?",
            (self._agent_id, ledger_key, action_class, self._window_start()),
        )
        return Money.from_cents(int(row["total"]) if row else 0)

    def state(
        self, ledger_key: str, action_class: str, pending: Money | None = None
    ) -> LedgerState:
        """Where this key stands, optionally including a call not yet made."""
        total = self.window_total(ledger_key, action_class) + (pending or Money.zero())
        threshold = self.threshold_for(action_class)
        return LedgerState(
            ledger_key=ledger_key,
            action_class=action_class,
            window_total=total,
            threshold=threshold,
            entry_count=self.entry_count(ledger_key, action_class),
            exceeded=threshold is not None and total > threshold,
        )

    def would_exceed(self, ledger_key: str, action_class: str, amount: Money) -> bool:
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
            "(agent_id, ledger_key, action_class, actor, amount_cents, trace_id, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self._agent_id,
                entry.ledger_key,
                entry.action_class,
                entry.actor,
                entry.amount.cents,
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
            LedgerEntry.model_validate(
                {
                    **{k: v for k, v in row.items() if k != "amount_cents"},
                    "amount": Money.from_cents(int(row["amount_cents"])),
                }
            )
            for row in self._store.iter_rows(
                "SELECT agent_id, ledger_key, action_class, actor, amount_cents, trace_id, recorded_at "
                "FROM platform_gatekeeper_ledger WHERE agent_id = ? AND ledger_key = ? "
                "AND action_class = ? AND recorded_at >= ? ORDER BY entry_id",
                (self._agent_id, ledger_key, action_class, self._window_start()),
            )
        ]

    def _window_start(self) -> str:
        """ISO timestamp marking the start of the rolling window."""
        return (utc_now() - self._window).isoformat()
