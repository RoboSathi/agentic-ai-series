"""Memory RBAC over the same permission model the agent's tools use.

Grants come from two places, and only two:

  * the platform's role defaults (`memory.governance.role_permissions`),
    so a customer may *propose* a preference and a curator may approve;
  * for the agent itself, its persisted `AgentPolicy` — the system of
    record for what this agent may do — which can widen its memory grants
    without a config change.

It implements `memory`'s `MemoryAuthorizer` port; `memory` never imports
this module, which is what keeps the storage layer free of `security`.
"""

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.memory.governance import MemoryDecision
from agent_platform.memory.types import MemoryPrincipal
from agent_platform.security.auth.permissions import permission_granted
from agent_platform.security.auth.policy import AgentPolicy


class PolicyMemoryAuthorizer:
    """Authorizes memory operations from role defaults plus the agent's policy."""

    def __init__(self, policy: AgentPolicy | None = None) -> None:
        settings = get_settings()
        self._role_grants = settings.memory.governance.role_permissions
        self._enabled = settings.security.auth.enabled
        self._policy = policy
        self._logger = get_logger("auth")

    def authorize(self, principal: MemoryPrincipal, permission: str) -> MemoryDecision:
        if not self._enabled:
            return MemoryDecision(allowed=True, permission=permission, reason="auth disabled")
        grant = permission_granted(self._grants_for(principal), permission)
        if grant is None:
            self._logger.warning(
                "Memory operation denied",
                actor=principal.actor,
                role=principal.role,
                permission=permission,
            )
            return MemoryDecision(
                allowed=False,
                permission=permission,
                reason=f"role '{principal.role}' holds no grant covering '{permission}'",
            )
        return MemoryDecision(
            allowed=True,
            permission=permission,
            reason=f"granted to '{principal.role}' by '{grant}'",
            granted_by=grant,
        )

    def _grants_for(self, principal: MemoryPrincipal) -> list[str]:
        grants = list(self._role_grants.get(principal.role, []))
        policy = self._policy
        if principal.role == "agent" and policy is not None and principal.actor == policy.agent_id:
            grants.extend(policy.permissions)
        return grants
