"""Small DB-backed runtime policy for one deployed agent.

Design-time autonomy decisions (Sheridan / Three Gates) are intentionally not
stored here. This module only loads runtime authority and operational limits.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agentic_commons.persistence.platform import PlatformStore, as_platform_store

_REQUIRED_COLUMNS = {
    "agent_id",
    "agent_name",
    "version",
    "enabled",
    "role",
    "authority_rank",
    "max_refund_amount",
    "refund_budget_limit",
    "refund_budget_window_hours",
    "created_at",
    "updated_at",
}


class AgentPolicy(BaseModel):
    """Immutable runtime policy snapshot for one deployed agent."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    role: str
    authority_rank: int = Field(ge=1)
    permissions: set[str] = Field(default_factory=set)
    max_refund_amount: float = Field(ge=0)
    refund_budget_limit: float = Field(ge=0)
    refund_budget_window_hours: int = Field(default=24, gt=0)
    enabled: bool = True
    updated_at: str


class AgentPolicyRepository:
    """Load the agent's runtime authority from the platform agent tables."""

    def __init__(self, store: PlatformStore) -> None:
        self._store = as_platform_store(store)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        columns = {
            str(row["name"])
            for row in self._store.iter_rows("PRAGMA table_info(platform_agents)")
        }
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise RuntimeError(
                "Customer Support agent database has an old platform_agents schema. "
                f"Database: '{self._store.db_path}'. Missing columns: {', '.join(missing)}. "
                "Run 'make dummy-data' to rebuild it."
            )

    def get(self, agent_id: str | None = None) -> AgentPolicy:
        target = agent_id or self._store.agent_id
        row = self._store.query_one(
            "SELECT agent_id, role, authority_rank, max_refund_amount, "
            "refund_budget_limit, refund_budget_window_hours, enabled, updated_at "
            "FROM platform_agents WHERE agent_id = ?",
            (target,),
        )
        if row is None:
            raise RuntimeError(f"No runtime policy is configured for agent '{target}'")

        permissions = {
            str(item["permission"])
            for item in self._store.iter_rows(
                "SELECT permission FROM platform_agent_permissions "
                "WHERE agent_id = ? ORDER BY permission",
                (target,),
            )
        }
        return AgentPolicy(
            agent_id=str(row["agent_id"]),
            role=str(row["role"]),
            authority_rank=int(row["authority_rank"]),
            permissions=permissions,
            max_refund_amount=float(row["max_refund_amount"]),
            refund_budget_limit=float(row["refund_budget_limit"]),
            refund_budget_window_hours=int(row["refund_budget_window_hours"]),
            enabled=bool(row["enabled"]),
            updated_at=str(row["updated_at"]),
        )

    def upsert(self, policy: AgentPolicy) -> AgentPolicy:
        if policy.agent_id != self._store.agent_id:
            raise ValueError(
                f"Policy belongs to '{policy.agent_id}', but this store is scoped to "
                f"'{self._store.agent_id}'"
            )
        self._store.execute(
            "UPDATE platform_agents SET role = ?, authority_rank = ?, "
            "max_refund_amount = ?, refund_budget_limit = ?, "
            "refund_budget_window_hours = ?, updated_at = ? WHERE agent_id = ?",
            (
                policy.role,
                policy.authority_rank,
                policy.max_refund_amount,
                policy.refund_budget_limit,
                policy.refund_budget_window_hours,
                policy.updated_at,
                policy.agent_id,
            ),
        )
        self._store.execute(
            "DELETE FROM platform_agent_permissions WHERE agent_id = ?",
            (policy.agent_id,),
        )
        if policy.permissions:
            self._store.execute_many(
                "INSERT INTO platform_agent_permissions (agent_id, permission) VALUES (?, ?)",
                [(policy.agent_id, permission) for permission in sorted(policy.permissions)],
            )
        return self.get(policy.agent_id)
