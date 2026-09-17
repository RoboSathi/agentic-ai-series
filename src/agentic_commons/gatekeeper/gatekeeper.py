"""Gatekeeper stub for Phase 0."""

from agentic_commons.gatekeeper.types import Action, AuthorizationDecision
from agentic_commons.logging import log_call


class Gatekeeper:
    """Gatekeeper interface stub."""

    @log_call("gatekeeper")
    def authorize(self, action: Action) -> AuthorizationDecision:
        """Phase 0 stub — no real authorization yet."""
        return AuthorizationDecision(authorized=False, reason="")
