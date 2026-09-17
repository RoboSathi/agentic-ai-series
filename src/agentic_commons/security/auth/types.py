"""Type definitions for auth security."""

from pydantic import BaseModel


class AuthResult(BaseModel):
    """Result of authentication."""

    authenticated: bool = False
    user_id: str = ""
