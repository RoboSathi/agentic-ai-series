-- Schema for the customer support agent's database.
--
-- order_status:  pending / shipped / delivered / cancelled
-- refund_status: none / pending_review / approved / denied / issued / completed
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
    -- Money is integer cents, never REAL (see agent_platform.core.primitives.money).
    amount_cents INTEGER NOT NULL,
    -- What was bought. `amount_cents` stays the order total; unit price
    -- times quantity is what makes it up.
    product_name TEXT NOT NULL DEFAULT '',
    product_sku TEXT NOT NULL DEFAULT '',
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price_cents INTEGER NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT 'USD',
    -- Where it is. Nullable on purpose, and the generator honours the
    -- same rule the agent does: a cancelled order has no ETA, and a
    -- delivered one has a delivered_at rather than a location in transit.
    carrier TEXT,
    tracking_number TEXT,
    estimated_delivery_date TEXT,
    delivered_at TEXT,
    current_location TEXT,
    -- The carrier's own note on the delivery: third-party text the agent
    -- reads and is never instructed by.
    carrier_note TEXT,
    -- When an `issued` refund settles; NULL in every other refund state.
    refund_settles_at TEXT
);

-- Every order lookup is scoped to the authenticated customer, so the
-- owning column is indexed.
CREATE INDEX IF NOT EXISTS idx_orders_user ON orders (user_id);

-- Escalations (a refund, a cancellation's refund, a reply the hallucination check cut) are
-- items in the platform's one review queue, `platform_review_queue` (ADR 0012).



-- Platform-owned runtime authority for this deployed agent.
CREATE TABLE IF NOT EXISTS platform_agents (
    agent_id                     TEXT PRIMARY KEY,
    agent_name                   TEXT NOT NULL,
    version                      TEXT NOT NULL,
    enabled                      INTEGER NOT NULL DEFAULT 1,
    role                         TEXT NOT NULL DEFAULT 'unconfigured',
    authority_rank               INTEGER NOT NULL DEFAULT 999,
    created_at                   TEXT NOT NULL,
    updated_at                   TEXT NOT NULL
);

-- What an agent may spend, per class of action. Keyed by action class
-- rather than columns named after one product's verbs, so this schema
-- describes any agent the platform runs.
CREATE TABLE IF NOT EXISTS platform_agent_budgets (
    agent_id            TEXT NOT NULL,
    action_class        TEXT NOT NULL,
    per_action_limit_cents INTEGER NOT NULL DEFAULT 0,
    window_limit_cents     INTEGER NOT NULL DEFAULT 0,
    window_hours        INTEGER NOT NULL DEFAULT 24,
    PRIMARY KEY (agent_id, action_class),
    FOREIGN KEY (agent_id) REFERENCES platform_agents(agent_id)
);

CREATE TABLE IF NOT EXISTS platform_agent_permissions (
    agent_id    TEXT NOT NULL,
    permission  TEXT NOT NULL,
    PRIMARY KEY (agent_id, permission),
    FOREIGN KEY (agent_id) REFERENCES platform_agents(agent_id)
);

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
