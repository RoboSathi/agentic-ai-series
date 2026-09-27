"""Domain repositories (LLD §21).

These satisfy `persistence.Repository` and hold the only SQL the agent
knows about. The platform stays schema-agnostic; the domain knowledge
lives here, behind methods whose names are the domain's vocabulary
rather than the table's.
"""

from typing import Any

from agentic_commons.logging import get_logger, logged
from agentic_commons.money import Money
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.domain.models import (
    HitlRequest,
    HitlStatus,
    Order,
    OrderStatus,
    RefundStatus,
    User,
)

# Listed once. Every query below is built from this, so a column added to
# the schema reaches `get`, `save`, `query`, `for_user` and the
# customer-scoped read together rather than one at a time.
_ORDER_COLUMNS = (
    "order_id, user_id, timestamp, order_status, refund_status, amount_cents, "
    "product_name, product_sku, quantity, unit_price_cents, currency, "
    "carrier, tracking_number, estimated_delivery_date, delivered_at, current_location"
)
_ORDER_COLUMN_NAMES = tuple(name.strip() for name in _ORDER_COLUMNS.split(","))
_ORDER_PLACEHOLDERS = ", ".join("?" for _ in _ORDER_COLUMN_NAMES)
# Money is stored as integer cents and held as `Money`; these columns are
# the only place the two meet.
_MONEY_COLUMNS = {"amount_cents": "amount", "unit_price_cents": "unit_price"}


def _order_from(row: dict[str, Any]) -> Order:
    """An order row, with its cents columns read back as `Money`."""
    values = {key: value for key, value in row.items() if key not in _MONEY_COLUMNS}
    for column, field in _MONEY_COLUMNS.items():
        values[field] = Money.from_cents(int(row[column]))
    return Order.model_validate(values)


def _order_values(order: Order) -> tuple[Any, ...]:
    """An order's values in column order, money as integer cents."""
    return tuple(
        getattr(order, _MONEY_COLUMNS[column]).cents
        if column in _MONEY_COLUMNS
        else getattr(order, column)
        for column in _ORDER_COLUMN_NAMES
    )


def _hitl_from(row: dict[str, Any]) -> HitlRequest:
    values = {key: value for key, value in row.items() if key != "requested_amount_cents"}
    values["requested_amount"] = Money.from_cents(int(row["requested_amount_cents"]))
    return HitlRequest.model_validate(values)
_USER_COLUMNS = "user_id, user_code, created_at, name, email, phone"


class OrdersRepository:
    """Orders, and the refund state transitions they can undergo."""

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self._pool = pool
        self._logger = get_logger("customer_support_agent")

    def get(self, id: int) -> Order | None:
        """One order by id."""
        row = self._pool.query_one(f"SELECT {_ORDER_COLUMNS} FROM orders WHERE order_id = ?", (id,))
        return _order_from(row) if row else None

    def save(self, entity: Order) -> None:
        """Insert or replace an order.

        The values are taken from the model in column order rather than
        listed by hand: a hand-written tuple is the thing that silently
        stops matching when a column is added.
        """
        self._pool.execute(
            f"INSERT OR REPLACE INTO orders ({_ORDER_COLUMNS}) VALUES ({_ORDER_PLACEHOLDERS})",
            _order_values(entity),
        )

    def query(self, **filters: Any) -> list[Order]:
        """Orders matching equality filters on any column."""
        if not filters:
            return [
                _order_from(row)
                for row in self._pool.iter_rows(f"SELECT {_ORDER_COLUMNS} FROM orders")
            ]
        clause = " AND ".join(f"{column} = ?" for column in filters)
        return [
            _order_from(row)
            for row in self._pool.iter_rows(
                f"SELECT {_ORDER_COLUMNS} FROM orders WHERE {clause}", tuple(filters.values())
            )
        ]

    def get_for_customer(self, order_id: int, customer_id: int | None) -> Order | None:
        """One order, but only if this customer owns it.

        Scoping happens in the query, not after it. Fetching the row and
        then comparing owners works until someone forgets the comparison;
        a query that cannot return another customer's row has nothing to
        forget. An empty customer id matches nothing, so an
        unauthenticated request sees no orders rather than all of them.
        """
        if customer_id is None:
            return None
        row = self._pool.query_one(
            f"SELECT {_ORDER_COLUMNS} FROM orders WHERE order_id = ? AND user_id = ?",
            (order_id, customer_id),
        )
        return _order_from(row) if row else None

    def for_user(self, user_id: int) -> list[Order]:
        """Every order belonging to one customer."""
        return self.query(user_id=user_id)

    @logged("customer_support_agent")
    def set_refund_status(
        self, order_id: int, status: RefundStatus, expected: RefundStatus | None = None
    ) -> bool:
        """Move an order to a new refund state.

        With `expected`, this is a compare-and-swap: the row moves only if
        it is still in the state the caller believed it was in, and the
        return value says whether that held. Without it, the write is
        unconditional.

        The conditional form is what makes a concurrent refund lose rather
        than double-pay. A blind `UPDATE ... WHERE order_id = ?` succeeds
        just as happily against a row another turn has already marked
        completed, and reports success either way.
        """
        if expected is None:
            result = self._pool.execute(
                "UPDATE orders SET refund_status = ? WHERE order_id = ?", (status, order_id)
            )
        else:
            result = self._pool.execute(
                "UPDATE orders SET refund_status = ? "
                "WHERE order_id = ? AND refund_status = ?",
                (status, order_id, expected),
            )
        return result.affected > 0

    @logged("customer_support_agent")
    def set_order_status(
        self, order_id: int, status: OrderStatus, expected: OrderStatus | None = None
    ) -> bool:
        """Move an order to a new lifecycle state; compare-and-swap with `expected`.

        Same contract as `set_refund_status`: a cancellation racing a
        shipment must lose cleanly, not overwrite "shipped" blindly.
        """
        if expected is None:
            result = self._pool.execute(
                "UPDATE orders SET order_status = ? WHERE order_id = ?", (status, order_id)
            )
        else:
            result = self._pool.execute(
                "UPDATE orders SET order_status = ? WHERE order_id = ? AND order_status = ?",
                (status, order_id, expected),
            )
        return result.affected > 0


class UsersRepository:
    """Customers."""

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self._pool = pool

    def get(self, id: int) -> User | None:
        """One customer by internal id."""
        row = self._pool.query_one(f"SELECT {_USER_COLUMNS} FROM users WHERE user_id = ?", (id,))
        return User.model_validate(row) if row else None

    def order_count_for(self, user_id: int) -> int:
        """How many orders a customer has — shown in the verified read-back."""
        row = self._pool.query_one("SELECT COUNT(*) AS n FROM orders WHERE user_id = ?", (user_id,))
        return int(row["n"]) if row else 0

    def get_by_code(self, user_code: str) -> User | None:
        """One customer by the code they typed.

        Matching is case-insensitive and ignores surrounding whitespace:
        a customer reading a code off an email should not fail
        authentication over capitalisation.
        """
        normalized = user_code.strip().upper()
        if not normalized:
            return None
        row = self._pool.query_one(
            f"SELECT {_USER_COLUMNS} FROM users WHERE user_code = ?", (normalized,)
        )
        return User.model_validate(row) if row else None

    def save(self, entity: User) -> None:
        """Insert or replace a customer."""
        self._pool.execute(
            f"INSERT OR REPLACE INTO users ({_USER_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            (
                entity.user_id,
                entity.user_code,
                entity.created_at,
                entity.name,
                entity.email,
                entity.phone,
            ),
        )

    def query(self, **filters: Any) -> list[User]:
        """Customers matching equality filters."""
        clause = " AND ".join(f"{column} = ?" for column in filters) or "1 = 1"
        return [
            User.model_validate(row)
            for row in self._pool.iter_rows(
                f"SELECT {_USER_COLUMNS} FROM users WHERE {clause}", tuple(filters.values())
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

    def get(self, id: int) -> HitlRequest | None:
        """One queue entry by id."""
        row = self._pool.query_one("SELECT * FROM hitl_queue WHERE id = ?", (id,))
        return _hitl_from(row) if row else None

    def save(self, entity: HitlRequest) -> None:
        """Enqueue a request for human review."""
        self._pool.execute(
            "INSERT INTO hitl_queue "
            "(order_id, user_id, requested_amount_cents, action, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                entity.order_id,
                entity.user_id,
                entity.requested_amount.cents,
                entity.action,
                entity.status,
                entity.created_at,
            ),
        )

    def query(self, **filters: Any) -> list[HitlRequest]:
        """Queue entries matching equality filters."""
        clause = " AND ".join(f"{column} = ?" for column in filters) or "1 = 1"
        return [
            _hitl_from(row)
            for row in self._pool.iter_rows(
                f"SELECT * FROM hitl_queue WHERE {clause} ORDER BY id", tuple(filters.values())
            )
        ]

    def pending(self) -> list[HitlRequest]:
        """Everything still awaiting a human decision."""
        return self.query(status="pending")

    def pending_for_order(self, order_id: int) -> HitlRequest | None:
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
        order_id   INTEGER,
        user_id    INTEGER NOT NULL,
        amount_cents INTEGER NOT NULL,
        reason     TEXT NOT NULL,
        reversed   INTEGER NOT NULL DEFAULT 0,
        issued_at  TEXT NOT NULL
    );
    """

    def __init__(self, pool: SqliteConnectionPool) -> None:
        self._pool = pool
        self._pool.execute_script(self._SCHEMA)

    def record(
        self, refund_id: str, order_id: int | None, user_id: int, amount: Money, reason: str
    ) -> None:
        """Record an issued refund, in integer cents."""
        self._pool.execute(
            "INSERT OR REPLACE INTO refunds (refund_id, order_id, user_id, amount_cents, reason, "
            "reversed, issued_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (refund_id, order_id, user_id, amount.cents, reason, utc_now().isoformat()),
        )

    def reverse(self, refund_id: str) -> bool:
        """Mark a refund reversed — the saga's compensation for this step."""
        result = self._pool.execute(
            "UPDATE refunds SET reversed = 1 WHERE refund_id = ?", (refund_id,)
        )
        return result.affected > 0

    def settled_for_order(self, order_id: int) -> Money:
        """Total actually paid out against one order, reversals excluded.

        The read-back half of effect verification: a refund is not what
        the tool said it did, it is what the ledger says was paid.
        """
        row = self._pool.query_one(
            "SELECT COALESCE(SUM(amount_cents), 0) AS total FROM refunds "
            "WHERE order_id = ? AND reversed = 0",
            (order_id,),
        )
        return Money.from_cents(int(row["total"]) if row else 0)

    def total_for_user(self, user_id: int) -> Money:
        """Total refunded to one customer, excluding reversals."""
        row = self._pool.query_one(
            "SELECT COALESCE(SUM(amount_cents), 0) AS total FROM refunds "
            "WHERE user_id = ? AND reversed = 0",
            (user_id,),
        )
        return Money.from_cents(int(row["total"]) if row else 0)
