"""Auth stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.security.auth.types import AuthResult


class Auth:
    """Authentication interface stub."""

    @log_call("security", toggle_path="security.auth.enabled")
    def authenticate(self, token: str) -> AuthResult:
        """Phase 0 stub — no real auth yet."""
        return AuthResult(authenticated=False, user_id="")
