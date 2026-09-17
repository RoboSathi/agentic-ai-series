"""Seed the SQLite database with sample users/orders for local testing.

Idempotent: re-running clears and re-inserts the same fixed sample rows
rather than accumulating duplicates.
"""

from dataclasses import dataclass

from agentic_commons.logging import get_logger
from agentic_commons.persistence.database import Database
from customer_support_agent.core.repositories import (
    OrdersRepository,
    OrderStatus,
    RefundStatus,
    UsersRepository,
)
from customer_support_agent.core.schema import ensure_schema


@dataclass
class _SeedUser:
    user_id: str
    email: str
    phone: str


@dataclass
class _SeedOrder:
    order_id: str
    user_id: str
    order_status: OrderStatus
    refund_status: RefundStatus
    amount: float


_USERS = [
    _SeedUser(user_id="user-1", email="alice@example.com", phone="555-0100"),
    _SeedUser(user_id="user-2", email="bob@example.com", phone="555-0101"),
]

_ORDERS = [
    _SeedOrder(
        order_id="12345",
        user_id="user-1",
        order_status="delivered",
        refund_status="none",
        amount=29.99,
    ),
    _SeedOrder(
        order_id="12346",
        user_id="user-1",
        order_status="delivered",
        refund_status="none",
        amount=149.50,
    ),
    _SeedOrder(
        order_id="12347",
        user_id="user-2",
        order_status="shipped",
        refund_status="none",
        amount=19.99,
    ),
    _SeedOrder(
        order_id="12348",
        user_id="user-2",
        order_status="delivered",
        refund_status="completed",
        amount=42.00,
    ),
]


def main() -> None:
    """Reset and reseed the database with a fixed set of sample rows."""
    logger = get_logger("persistence")
    database = Database()

    database.execute("DROP TABLE IF EXISTS hitl_queue")
    database.execute("DROP TABLE IF EXISTS orders")
    database.execute("DROP TABLE IF EXISTS users")
    ensure_schema(database)

    users = UsersRepository(database)
    orders = OrdersRepository(database)

    for user in _USERS:
        users.insert_user(user.user_id, user.email, user.phone)
    for order in _ORDERS:
        orders.insert_order(
            order.order_id,
            order.user_id,
            order.order_status,
            order.refund_status,
            order.amount,
        )

    logger.info(
        f"Seeded {len(_USERS)} users and {len(_ORDERS)} orders "
        f"(order ids: {', '.join(o.order_id for o in _ORDERS)})"
    )


if __name__ == "__main__":
    main()
