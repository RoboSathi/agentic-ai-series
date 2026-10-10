"""Authorization value types (LLD §12.1)."""

from pydantic import BaseModel, Field

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


class AccessDecision(BaseModel):
    """Whether a principal may perform an action, and on what basis."""

    allowed: bool
    actor: str
    action: str
    permission: Permission = ""
    reason: str = ""
    via_elevation: bool = False
    role: str = ""
