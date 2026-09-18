"""Domain repositories (LLD §21).

These satisfy `persistence.Repository` and hold the only SQL the agent
knows about. The platform stays schema-agnostic; the domain knowledge
lives here, behind methods whose names are the domain's vocabulary
rather than the table's.
"""

from datetime import timedelta
from typing import Any

from agentic_commons.logging import get_logger, logged
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.core.schema import (
    HitlRequest,
    HitlStatus,
    Order,
    RefundStatus,
    User,
)

_ORDER_COLUMNS = "order_id, user_id, timestamp, order_status, refund_status, amount"


class OrdersRepository:
    """Orders, and the refund state transitions they can undergo."""

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self._pool = pool
        self._logger = get_logger("customer_support_agent")

    def get(self, id: str) -> Order | None:
        """One order by id."""
        row = self._pool.query_one(f"SELECT {_ORDER_COLUMNS} FROM orders WHERE order_id = ?", (id,))
        return Order.model_validate(row) if row else None

    def save(self, entity: Order) -> None:
        """Insert or replace an order."""
        self._pool.execute(
            f"INSERT OR REPLACE INTO orders ({_ORDER_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            (
                entity.order_id,
                entity.user_id,
                entity.timestamp,
                entity.order_status,
                entity.refund_status,
                entity.amount,
            ),
        )

    def query(self, **filters: Any) -> list[Order]:
        """Orders matching equality filters on any column."""
        if not filters:
            return [
                Order.model_validate(row)
                for row in self._pool.iter_rows(f"SELECT {_ORDER_COLUMNS} FROM orders")
            ]
        clause = " AND ".join(f"{column} = ?" for column in filters)
        return [
            Order.model_validate(row)
            for row in self._pool.iter_rows(
                f"SELECT {_ORDER_COLUMNS} FROM orders WHERE {clause}", tuple(filters.values())
            )
        ]

    def get_for_customer(self, order_id: str, customer_id: str) -> Order | None:
        """One order, but only if this customer owns it.

        Scoping happens in the query, not after it. Fetching the row and
        then comparing owners works until someone forgets the comparison;
        a query that cannot return another customer's row has nothing to
        forget. An empty customer id matches nothing, so an
        unauthenticated request sees no orders rather than all of them.
        """
        if not customer_id:
            return None
        row = self._pool.query_one(
            f"SELECT {_ORDER_COLUMNS} FROM orders WHERE order_id = ? AND user_id = ?",
            (order_id, customer_id),
        )
        return Order.model_validate(row) if row else None

    def for_user(self, user_id: str) -> list[Order]:
        """Every order belonging to one customer."""
        return self.query(user_id=user_id)

    @logged("customer_support_agent")
    def set_refund_status(self, order_id: str, status: RefundStatus) -> bool:
        """Move an order to a new refund state, reporting whether it existed."""
        result = self._pool.execute(
            "UPDATE orders SET refund_status = ? WHERE order_id = ?", (status, order_id)
        )
        return result.affected > 0


class UsersRepository:
    """Customers."""

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self._pool = pool

    def get(self, id: str) -> User | None:
        """One customer by id."""
        row = self._pool.query_one(
            "SELECT user_id, created_at, email, phone FROM users WHERE user_id = ?", (id,)
        )
        return User.model_validate(row) if row else None

    def save(self, entity: User) -> None:
        """Insert or replace a customer."""
        self._pool.execute(
            "INSERT OR REPLACE INTO users (user_id, created_at, email, phone) VALUES (?, ?, ?, ?)",
            (entity.user_id, entity.created_at, entity.email, entity.phone),
        )

    def query(self, **filters: Any) -> list[User]:
        """Customers matching equality filters."""
        clause = " AND ".join(f"{column} = ?" for column in filters) or "1 = 1"
        return [
            User.model_validate(row)
            for row in self._pool.iter_rows(
                f"SELECT user_id, created_at, email, phone FROM users WHERE {clause}",
                tuple(filters.values()),
            )
        ]


class HitlQueueRepository:
    """The human-in-the-loop queue.

    Append-only in spirit and in surface class: entries are added and
    their status is decided by a human. The agent may enqueue; it may
    never approve.
    """

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self._pool = pool
        self._logger = get_logger("customer_support_agent")

    def get(self, id: str) -> HitlRequest | None:
        """One queue entry by id."""
        row = self._pool.query_one("SELECT * FROM hitl_queue WHERE id = ?", (id,))
        return HitlRequest.model_validate(row) if row else None

    def save(self, entity: HitlRequest) -> None:
        """Enqueue a request for human review."""
        self._pool.execute(
            "INSERT INTO hitl_queue (order_id, user_id, requested_amount, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                entity.order_id,
                entity.user_id,
                entity.requested_amount,
                entity.status,
                entity.created_at,
            ),
        )

    def query(self, **filters: Any) -> list[HitlRequest]:
        """Queue entries matching equality filters."""
        clause = " AND ".join(f"{column} = ?" for column in filters) or "1 = 1"
        return [
            HitlRequest.model_validate(row)
            for row in self._pool.iter_rows(
                f"SELECT * FROM hitl_queue WHERE {clause} ORDER BY id", tuple(filters.values())
            )
        ]

    def pending(self) -> list[HitlRequest]:
        """Everything still awaiting a human decision."""
        return self.query(status="pending")

    def pending_for_order(self, order_id: str) -> HitlRequest | None:
        """An existing pending request for an order, if there is one.

        Checked before enqueuing so a customer who asks twice does not
        produce two queue entries and two refunds for one order.
        """
        matches = self.query(order_id=order_id, status="pending")
        return matches[0] if matches else None

    @logged("customer_support_agent")
    def decide(self, request_id: int, status: HitlStatus) -> bool:
        """Record a human's decision on a queued request."""
        result = self._pool.execute(
            "UPDATE hitl_queue SET status = ? WHERE id = ? AND status = 'pending'",
            (status, request_id),
        )
        return result.affected > 0

    def remove(self, request_id: int) -> None:
        """Remove a queue entry — the compensation for a failed enqueue."""
        self._pool.execute("DELETE FROM hitl_queue WHERE id = ?", (request_id,))

    def last_id(self) -> int:
        """The most recently inserted queue id."""
        row = self._pool.query_one("SELECT MAX(id) AS id FROM hitl_queue")
        return int(row["id"]) if row and row["id"] is not None else 0


class RefundLedgerRepository:
    """Completed refunds, recorded separately from the order they refunded.

    Two records, on purpose. The order's `refund_status` says what is
    true now; this ledger says what happened and when. Reconstructing the
    second from the first is impossible once a status has changed twice.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS refunds (
        refund_id  TEXT PRIMARY KEY,
        order_id   TEXT NOT NULL,
        user_id    TEXT NOT NULL,
        amount     REAL NOT NULL,
        reason     TEXT NOT NULL,
        reversed   INTEGER NOT NULL DEFAULT 0,
        issued_at  TEXT NOT NULL
    );
    """

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self._pool = pool
        self._pool.execute_script(self._SCHEMA)

    def record(
        self, refund_id: str, order_id: str, user_id: str, amount: float, reason: str
    ) -> None:
        """Record an issued refund."""
        self._pool.execute(
            "INSERT OR REPLACE INTO refunds (refund_id, order_id, user_id, amount, reason, "
            "reversed, issued_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (refund_id, order_id, user_id, amount, reason, utc_now().isoformat()),
        )

    def reverse(self, refund_id: str) -> bool:
        """Mark a refund reversed — the saga's compensation for this step."""
        result = self._pool.execute(
            "UPDATE refunds SET reversed = 1 WHERE refund_id = ?", (refund_id,)
        )
        return result.affected > 0

    def goodwill_total(self, user_id: str, window_days: int) -> float:
        """Goodwill credited to one customer inside a rolling window.

        Windowed rather than lifetime: a cap with no window eventually
        blocks every long-standing customer, and one with no cap at all
        is not a control.
        """
        cutoff = (utc_now() - timedelta(days=window_days)).isoformat()
        row = self._pool.query_one(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM refunds "
            "WHERE user_id = ? AND reversed = 0 AND reason LIKE 'goodwill%' "
            "AND issued_at >= ?",
            (user_id, cutoff),
        )
        return float(row["total"]) if row else 0.0

    def total_for_user(self, user_id: str) -> float:
        """Total refunded to one customer, excluding reversals."""
        row = self._pool.query_one(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM refunds "
            "WHERE user_id = ? AND reversed = 0",
            (user_id,),
        )
        return float(row["total"]) if row else 0.0
