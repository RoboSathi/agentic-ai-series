"""RBAC and just-in-time privilege elevation (LLD §12.1)."""

from agentic_commons.security.auth.auth import (
    Authorizer,
    JustInTimeElevator,
    PrivilegeElevator,
    RBACRoleStore,
    RoleStore,
)
from agentic_commons.security.auth.types import (
    AccessDecision,
    ElevationGrant,
    Permission,
    Principal,
)

__all__ = [
    "AccessDecision",
    "Authorizer",
    "ElevationGrant",
    "JustInTimeElevator",
    "Permission",
    "Principal",
    "PrivilegeElevator",
    "RBACRoleStore",
    "RoleStore",
]
