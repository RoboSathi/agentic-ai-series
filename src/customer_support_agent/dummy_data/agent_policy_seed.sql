-- Runtime authority for the demo agent. Generated data must never change
-- design-time autonomy decisions; those live in the graph/ADR.

INSERT INTO platform_agents (
    agent_id, agent_name, version, enabled, role, authority_rank,
    max_refund_amount, refund_budget_limit, refund_budget_window_hours,
    created_at, updated_at
) VALUES (
    '__AGENT_ID__', '__AGENT_ID__', '__AGENT_VERSION__', 1, 'support_agent', 4,
    50.00, 4000.00, 24, datetime('now'), datetime('now')
)
ON CONFLICT(agent_id) DO UPDATE SET
    agent_name = excluded.agent_name,
    version = excluded.version,
    role = excluded.role,
    authority_rank = excluded.authority_rank,
    max_refund_amount = excluded.max_refund_amount,
    refund_budget_limit = excluded.refund_budget_limit,
    refund_budget_window_hours = excluded.refund_budget_window_hours,
    updated_at = excluded.updated_at;

DELETE FROM platform_agent_permissions WHERE agent_id = '__AGENT_ID__';

INSERT INTO platform_agent_permissions (agent_id, permission) VALUES
('__AGENT_ID__', 'read_orders'),
('__AGENT_ID__', 'process_refund'),
('__AGENT_ID__', 'queue_for_approval');
