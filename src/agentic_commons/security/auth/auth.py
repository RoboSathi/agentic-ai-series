"""Runtime authorization against the persisted policy of one agent."""

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, logged
from agentic_commons.security.auth.policy import AgentPolicy
from agentic_commons.security.auth.types import AccessDecision, Principal


class Authorizer:
    """Authorize an action using one trusted, persisted agent policy snapshot."""

    def __init__(self, policy: AgentPolicy | None = None) -> None:
        self._policy = policy
        self._enabled = get_settings().security.auth.enabled
        self._logger = get_logger("auth")

    @logged("auth")
    def authorize(
        self,
        principal: Principal,
        action: str,
        required_permission: str | None = None,
    ) -> AccessDecision:
        """Check identity and the permission declared by the trusted tool definition."""
        if not self._enabled:
            return AccessDecision(
                allowed=True,
                actor=principal.actor,
                action=action,
                permission=required_permission or "",
                reason="auth disabled",
            )

        policy = self._policy
        if policy is None:
            return AccessDecision(
                allowed=False,
                actor=principal.actor,
                action=action,
                permission=required_permission or "",
                reason="no persisted agent policy is loaded",
            )

        if principal.actor != policy.agent_id:
            return AccessDecision(
                allowed=False,
                actor=principal.actor,
                action=action,
                permission=required_permission or "",
                role=policy.role,
                reason="actor does not match the agent identity",
            )

        if not policy.enabled:
            return AccessDecision(
                allowed=False,
                actor=principal.actor,
                action=action,
                permission=required_permission or "",
                role=policy.role,
                reason="agent is disabled by persisted platform state",
            )

        if not required_permission:
            return AccessDecision(
                allowed=False,
                actor=principal.actor,
                action=action,
                role=policy.role,
                reason="action has no trusted design-time permission mapping",
            )

        if required_permission in policy.permissions:
            return AccessDecision(
                allowed=True,
                actor=principal.actor,
                action=action,
                permission=required_permission,
                role=policy.role,
                reason=f"permission granted to role '{policy.role}' by agent policy",
            )

        return AccessDecision(
            allowed=False,
            actor=principal.actor,
            action=action,
            permission=required_permission,
            role=policy.role,
            reason=f"role '{policy.role}' does not have '{required_permission}'",
        )
