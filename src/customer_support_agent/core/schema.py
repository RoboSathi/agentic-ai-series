"""SQLite schema for the customer support agent's domain tables.

order_status: pending / shipped / delivered / cancelled
refund_status: none / pending_review / approved / denied / completed
hitl_queue.status: pending / approved / denied
"""

from agentic_commons.persistence.database import Database

_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT
)
"""

_CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    timestamp TEXT NOT NULL,
    order_status TEXT NOT NULL,
    refund_status TEXT NOT NULL,
    amount REAL NOT NULL
)
"""

_CREATE_HITL_QUEUE = """
CREATE TABLE IF NOT EXISTS hitl_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    user_id TEXT NOT NULL,
    requested_amount REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def ensure_schema(database: Database) -> None:
    """Create the users/orders/hitl_queue tables if they don't already exist."""
    database.execute(_CREATE_USERS)
    database.execute(_CREATE_ORDERS)
    database.execute(_CREATE_HITL_QUEUE)
