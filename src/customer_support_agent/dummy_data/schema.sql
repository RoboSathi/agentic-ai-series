-- Schema for the customer support agent's database.
--
-- order_status:  pending / shipped / delivered / cancelled
-- refund_status: none / pending_review / approved / denied / completed
-- hitl_queue.status: pending / approved / denied
--
-- Ids are INTEGER: joins across these tables are on the id columns, and
-- SQLite compares integers directly rather than collating text.
--
-- user_code is how a customer authenticates. It is separate from user_id
-- on purpose: the id is an internal key that appears in joins and logs,
-- while the code is a credential the customer types. Conflating them
-- would make an enumerable primary key into the thing that grants access.
--
-- Single source of truth for the DDL: read by generate_dummy_data.py to
-- build agentic.db, and by customer_support_agent.core.schema.ensure_schema
-- to initialize any fresh database (e.g. in tests) at runtime.

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    user_code TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT
);

-- Authentication looks a customer up by code on every unauthenticated
-- turn, so that lookup is indexed rather than a scan.
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_code ON users (user_code);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    timestamp TEXT NOT NULL,
    order_status TEXT NOT NULL,
    refund_status TEXT NOT NULL,
    amount REAL NOT NULL
);

-- Every order lookup is scoped to the authenticated customer, so the
-- owning column is indexed.
CREATE INDEX IF NOT EXISTS idx_orders_user ON orders (user_id);

CREATE TABLE IF NOT EXISTS hitl_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    requested_amount REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hitl_status ON hitl_queue (status);
