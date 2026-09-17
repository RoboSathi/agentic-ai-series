-- Schema for the customer support agent's database.
--
-- order_status:  pending / shipped / delivered / cancelled
-- refund_status: none / pending_review / approved / denied / completed
-- hitl_queue.status: pending / approved / denied
--
-- Single source of truth for the DDL: read by generate_dummy_data.py to
-- build agentic.db, and by customer_support_agent.core.schema.ensure_schema
-- to initialize any fresh database (e.g. in tests) at runtime.

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    timestamp TEXT NOT NULL,
    order_status TEXT NOT NULL,
    refund_status TEXT NOT NULL,
    amount REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS hitl_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    user_id TEXT NOT NULL,
    requested_amount REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
