"""Type definitions for gatekeeper module."""

from pydantic import BaseModel


class Action(BaseModel):
    """An action to authorize."""

    name: str = ""


class AuthorizationDecision(BaseModel):
    """Authorization decision."""

    authorized: bool = False
    reason: str = ""
