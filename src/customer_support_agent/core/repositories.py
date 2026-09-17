"""Repositories for the customer support agent's domain tables.

These wrap the generic agentic_commons.persistence.Database with the
customer-support-specific queries the tools need — the platform stays
provider/schema-agnostic, the domain knowledge lives here.
"""

from datetime import datetime, timezone
from typing import Literal, TypedDict

from agentic_commons.persistence.database import Database

OrderStatus = Literal["pending", "shipped", "delivered", "cancelled"]
RefundStatus = Literal["none", "pending_review", "approved", "denied", "completed"]
HitlStatus = Literal["pending", "approved", "denied"]


class OrderRow(TypedDict):
    """One row of the orders table."""

    order_id: str
    user_id: str
    timestamp: str
    order_status: OrderStatus
    refund_status: RefundStatus
    amount: float


class UserRow(TypedDict):
    """One row of the users table."""

    user_id: str
    created_at: str
    email: str
    phone: str | None


class OrdersRepository:
    """Queries and mutations for the orders table."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def get_order(self, order_id: str) -> OrderRow | None:
        """Look up one order by id, or None if it doesn't exist."""
        result = self._database.execute(
            "SELECT order_id, user_id, timestamp, order_status, refund_status, amount "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        )
        return result.rows[0] if result.rows else None  # type: ignore[return-value]

    def set_refund_status(self, order_id: str, refund_status: RefundStatus) -> None:
        """Update the refund_status of one order."""
        self._database.execute(
            "UPDATE orders SET refund_status = ? WHERE order_id = ?",
            (refund_status, order_id),
        )

    def insert_order(
        self,
        order_id: str,
        user_id: str,
        order_status: OrderStatus,
        refund_status: RefundStatus,
        amount: float,
    ) -> None:
        """Insert one order row, timestamped now."""
        self._database.execute(
            "INSERT INTO orders "
            "(order_id, user_id, timestamp, order_status, refund_status, amount) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                order_id,
                user_id,
                datetime.now(timezone.utc).isoformat(),
                order_status,
                refund_status,
                amount,
            ),
        )


class UsersRepository:
    """Queries and mutations for the users table."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def get_user(self, user_id: str) -> UserRow | None:
        """Look up one user by id, or None if it doesn't exist."""
        result = self._database.execute(
            "SELECT user_id, created_at, email, phone FROM users WHERE user_id = ?",
            (user_id,),
        )
        return result.rows[0] if result.rows else None  # type: ignore[return-value]

    def insert_user(self, user_id: str, email: str, phone: str | None = None) -> None:
        """Insert one user row, timestamped now."""
        self._database.execute(
            "INSERT INTO users (user_id, created_at, email, phone) VALUES (?, ?, ?, ?)",
            (user_id, datetime.now(timezone.utc).isoformat(), email, phone),
        )


class HitlQueueRepository:
    """Queries and mutations for the hitl_queue table."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def enqueue(self, order_id: str, user_id: str, requested_amount: float) -> None:
        """Queue a refund for human approval."""
        self._database.execute(
            "INSERT INTO hitl_queue "
            "(order_id, user_id, requested_amount, status, created_at) "
            "VALUES (?, ?, ?, 'pending', ?)",
            (order_id, user_id, requested_amount, datetime.now(timezone.utc).isoformat()),
        )
