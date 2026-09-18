"""RBAC and just-in-time privilege elevation (LLD §12.1).

Two ideas, deliberately separated. RBAC answers "what is this principal
normally allowed to do?" — a standing, reviewable answer. Elevation
answers "may they do this one extra thing, right now, for this long?" —
an exceptional, expiring answer. Systems that only have the first grow
over-broad roles, because the only way to permit an exception is to widen
the role permanently.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, logged
from agentic_commons.observability.types import utc_now
from agentic_commons.security.auth.types import (
    AccessDecision,
    ElevationGrant,
    Permission,
    Principal,
)


@runtime_checkable
class RoleStore(Protocol):
    """Resolves the permissions a role carries."""

    def permissions_for(self, role: str) -> set[Permission]: ...


@runtime_checkable
class PrivilegeElevator(Protocol):
    """Issues scoped, time-boxed grants above a principal's standing roles."""

    def elevate(self, actor: str, permission: Permission, ttl_seconds: int) -> ElevationGrant: ...


class RBACRoleStore:
    """Roles and their permissions, from config (NIST RBAC).

    Role definitions are configuration because they are a policy artifact
    a human reviews and changes — not a constant an engineer compiles in.
    """

    def __init__(self) -> None:
        settings = get_settings().security.auth
        self._roles = {role: set(perms) for role, perms in settings.roles.items()}
        self._action_permissions = settings.action_permissions
        self._logger = get_logger("security")

    def permissions_for(self, role: str) -> set[Permission]:
        """Permissions carried by one role; empty for an unknown role."""
        if role not in self._roles:
            self._logger.warning("Unknown role requested", role=role)
        return set(self._roles.get(role, set()))

    def permissions_for_principal(self, principal: Principal) -> set[Permission]:
        """Union of the permissions across a principal's roles."""
        return set().union(*(self.permissions_for(role) for role in principal.roles)) or set()

    def permission_for_action(self, action: str) -> Permission | None:
        """The permission an action requires.

        An action absent from the mapping returns None, which callers must
        treat as deny: an unmapped action is an ungoverned one, and
        defaulting it to allowed is how capabilities leak.
        """
        bare = action.rsplit(".", maxsplit=1)[-1]
        return self._action_permissions.get(bare)

    def roles_granting(self, permission: Permission) -> list[str]:
        """Every role carrying a permission — the access-review query."""
        return sorted(role for role, perms in self._roles.items() if permission in perms)


class JustInTimeElevator:
    """Scoped, expiring grants, capped by the configured maximum TTL.

    The cap is enforced here rather than trusted from the caller: a
    request for a twelve-hour elevation is silently clamped, because the
    point of the mechanism is that it expires.
    """

    def __init__(self) -> None:
        self._max_ttl = get_settings().security.auth.elevation_max_ttl_seconds
        self._grants: dict[str, list[ElevationGrant]] = {}
        self._logger = get_logger("security")

    @logged("security")
    def elevate(
        self, actor: str, permission: Permission, ttl_seconds: int, reason: str = ""
    ) -> ElevationGrant:
        """Grant one permission to one actor for a bounded time."""
        capped = min(ttl_seconds, self._max_ttl)
        if capped < ttl_seconds:
            self._logger.info(
                "Elevation TTL capped", actor=actor, requested=ttl_seconds, granted=capped
            )
        grant = ElevationGrant(
            actor=actor, permission=permission, ttl_seconds=capped, reason=reason
        )
        self._grants.setdefault(actor, []).append(grant)
        return grant

    def active_permissions(self, actor: str, now: datetime | None = None) -> set[Permission]:
        """Permissions currently granted to an actor by live elevations."""
        moment = now or utc_now()
        return {
            grant.permission for grant in self._grants.get(actor, []) if grant.is_active(moment)
        }

    def revoke(self, actor: str, permission: Permission) -> None:
        """Drop a grant before it expires."""
        self._grants[actor] = [
            grant for grant in self._grants.get(actor, []) if grant.permission != permission
        ]

    def grants_for(self, actor: str) -> list[ElevationGrant]:
        """Every grant ever issued to an actor — the elevation audit view."""
        return list(self._grants.get(actor, []))


class Authorizer:
    """Single entry point for "may this principal do this?" (Facade).

    The Gatekeeper's second question — *who is asking?* — resolves here.
    This class answers only that: it has no view on blast radius, surface
    class or cumulative spend, which belong to the Gatekeeper proper.
    """

    def __init__(
        self,
        role_store: RBACRoleStore | None = None,
        elevator: JustInTimeElevator | None = None,
    ) -> None:
        self._roles = role_store or RBACRoleStore()
        self._elevator = elevator or JustInTimeElevator()
        self._enabled = get_settings().security.auth.enabled
        self._logger = get_logger("security")

    @property
    def elevator(self) -> JustInTimeElevator:
        """The elevation service, for callers that need to grant."""
        return self._elevator

    @logged("security")
    def authorize(self, principal: Principal, action: str) -> AccessDecision:
        """Decide whether a principal may perform an action."""
        if not self._enabled:
            return AccessDecision(
                allowed=True, actor=principal.actor, action=action, reason="auth disabled"
            )
        permission = self._roles.permission_for_action(action)
        if permission is None:
            return AccessDecision(
                allowed=False,
                actor=principal.actor,
                action=action,
                reason="action has no mapped permission",
            )
        if permission in self._roles.permissions_for_principal(principal):
            return AccessDecision(
                allowed=True,
                actor=principal.actor,
                action=action,
                permission=permission,
                reason="granted by role",
            )
        if permission in self._elevator.active_permissions(principal.actor):
            return AccessDecision(
                allowed=True,
                actor=principal.actor,
                action=action,
                permission=permission,
                reason="granted by active elevation",
                via_elevation=True,
            )
        return AccessDecision(
            allowed=False,
            actor=principal.actor,
            action=action,
            permission=permission,
            reason=f"no role of {principal.roles} carries '{permission}'",
        )

    def permitted_actions(self, principal: Principal) -> list[str]:
        """Every action a principal may currently perform (least-privilege review)."""
        held = self._roles.permissions_for_principal(principal) | self._elevator.active_permissions(
            principal.actor
        )
        return sorted(
            action
            for action in get_settings().security.auth.action_permissions
            if self._roles.permission_for_action(action) in held
        )
