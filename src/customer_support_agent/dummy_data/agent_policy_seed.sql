-- Runtime authority for the demo agent. Generated data must never change
-- design-time autonomy decisions; those live in the graph/ADR.

INSERT INTO platform_agents (
    agent_id, agent_name, version, enabled, role, authority_rank,
    created_at, updated_at
) VALUES (
    '__AGENT_ID__', '__AGENT_ID__', '__AGENT_VERSION__', 1, 'support_agent', 4,
    datetime('now'), datetime('now')
)
ON CONFLICT(agent_id) DO UPDATE SET
    agent_name = excluded.agent_name,
    version = excluded.version,
    role = excluded.role,
    authority_rank = excluded.authority_rank,
    updated_at = excluded.updated_at;

DELETE FROM platform_agent_permissions WHERE agent_id = '__AGENT_ID__';

INSERT INTO platform_agent_permissions (agent_id, permission) VALUES
('__AGENT_ID__', 'read_orders'),
('__AGENT_ID__', 'process_refund'),
('__AGENT_ID__', 'cancel_order');

-- What this agent may spend, per class of action.
--
-- `refund` is this agent's word, not the platform's: the budget table is
-- keyed by action class so an agent with three budgeted actions, or none,
-- needs no change to the shared schema.
--
-- Amounts are integer cents (never REAL — see agent_platform.primitives.money).
-- This row is the one place the refund limit lives. Nothing the model
-- sees states it; `make refund-limit <amount>` changes it.
-- per_action — above this one refund (or cancellation) escalates to a human.
-- window     — the running total over window_hours: the structuring defence
--              (E02). Many individually-permitted refunds are not many
--              permitted actions; they are one unpermitted large one.
DELETE FROM platform_agent_budgets WHERE agent_id = '__AGENT_ID__';

INSERT INTO platform_agent_budgets (
    agent_id, action_class, per_action_limit_cents, window_limit_cents, window_hours
) VALUES
('__AGENT_ID__', 'refund', 5000, 400000, 24);
