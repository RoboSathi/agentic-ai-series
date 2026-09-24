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
    "created_at",
    "updated_at",
}

# What an agent may spend on one class of action. Any agent can have any
# number of these; the platform does not know what the classes are called.
UNLIMITED = float("inf")


class ActionLimit(BaseModel):
    """What one class of action may cost this agent.

    `per_action` bounds a single proposal; `window` bounds the running
    total inside `window_hours`, which is what defeats a structuring
    attack — eighty individually-permitted actions summing to one that
    was never permitted.
    """

    model_config = ConfigDict(frozen=True)

    action_class: str
    per_action: float = Field(ge=0)
    window: float = Field(ge=0)
    window_hours: int = Field(default=24, gt=0)


class AgentPolicy(BaseModel):
    """Immutable runtime policy snapshot for one deployed agent."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    role: str
    authority_rank: int = Field(ge=1)
    permissions: set[str] = Field(default_factory=set)
    # Keyed by action class. An agent with no budgeted actions has an
    # empty map, which is a perfectly ordinary agent — not a broken one.
    limits: dict[str, ActionLimit] = Field(default_factory=dict)
    enabled: bool = True
    updated_at: str

    def limit_for(self, action_class: str) -> ActionLimit | None:
        """The limit declared for one action class, if there is one."""
        return self.limits.get(action_class)

    def per_action_limit(self, action_class: str) -> float:
        """The single-proposal ceiling, or unlimited when none is declared.

        Unlimited rather than zero on purpose: an undeclared budget means
        this action is not financially bounded, and defaulting to zero
        would silently forbid every action nobody thought to budget.
        Whether the action is *allowed at all* is the permission check's
        question, not this one's.
        """
        limit = self.limits.get(action_class)
        return limit.per_action if limit is not None else UNLIMITED


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
                "This database has an old platform_agents schema. "
                f"Database: '{self._store.db_path}'. Missing columns: {', '.join(missing)}. "
                "Rebuild it from the agent's own seed data."
            )

    def get(self, agent_id: str | None = None) -> AgentPolicy:
        target = agent_id or self._store.agent_id
        row = self._store.query_one(
            "SELECT agent_id, role, authority_rank, enabled, updated_at "
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
            limits=self._limits(target),
            enabled=bool(row["enabled"]),
            updated_at=str(row["updated_at"]),
        )

    def _limits(self, agent_id: str) -> dict[str, ActionLimit]:
        """Every budgeted action class declared for this agent."""
        return {
            str(row["action_class"]): ActionLimit(
                action_class=str(row["action_class"]),
                per_action=float(row["per_action_limit"]),
                window=float(row["window_limit"]),
                window_hours=int(row["window_hours"]),
            )
            for row in self._store.iter_rows(
                "SELECT action_class, per_action_limit, window_limit, window_hours "
                "FROM platform_agent_budgets WHERE agent_id = ? ORDER BY action_class",
                (agent_id,),
            )
        }

    def upsert(self, policy: AgentPolicy) -> AgentPolicy:
        if policy.agent_id != self._store.agent_id:
            raise ValueError(
                f"Policy belongs to '{policy.agent_id}', but this store is scoped to "
                f"'{self._store.agent_id}'"
            )
        self._store.execute(
            "UPDATE platform_agents SET role = ?, authority_rank = ?, "
            "updated_at = ? WHERE agent_id = ?",
            (policy.role, policy.authority_rank, policy.updated_at, policy.agent_id),
        )
        self._store.execute(
            "DELETE FROM platform_agent_budgets WHERE agent_id = ?", (policy.agent_id,)
        )
        if policy.limits:
            self._store.execute_many(
                "INSERT INTO platform_agent_budgets (agent_id, action_class, "
                "per_action_limit, window_limit, window_hours) VALUES (?, ?, ?, ?, ?)",
                [
                    (policy.agent_id, limit.action_class, limit.per_action,
                     limit.window, limit.window_hours)
                    for limit in sorted(policy.limits.values(), key=lambda item: item.action_class)
                ],
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
