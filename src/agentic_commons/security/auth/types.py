"""Authorization value types (LLD §12.1)."""

from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from agentic_commons.observability.types import utc_now

# A permission is a capability name, not a role — roles are bundles of these.
Permission = str


class Principal(BaseModel):
    """Who is asking, and what standing roles they hold.

    Roles are what a principal *is*; permissions are what they may do.
    Keeping them apart is what makes least privilege reviewable: you can
    read a role definition and say whether it is too broad.
    """

    actor: str
    roles: list[str] = Field(default_factory=list)


class ElevationGrant(BaseModel):
    """A scoped, time-boxed privilege grant (never a standing elevation)."""

    actor: str
    permission: Permission
    granted_at: datetime = Field(default_factory=utc_now)
    ttl_seconds: int = 0
    reason: str = ""

    @property
    def expires_at(self) -> datetime:
        """When this grant stops being honoured."""
        return self.granted_at + timedelta(seconds=self.ttl_seconds)

    def is_active(self, now: datetime | None = None) -> bool:
        """Whether the grant is still within its TTL."""
        return (now or utc_now()) < self.expires_at


class AccessDecision(BaseModel):
    """Whether a principal may perform an action, and on what basis."""

    allowed: bool
    actor: str
    action: str
    permission: Permission = ""
    reason: str = ""
    via_elevation: bool = False
    role: str = ""
