"""Idempotency guard stub for Phase 0."""

from agentic_commons.logging import log_call


class IdempotencyGuard:
    """Idempotency guard stub."""

    @log_call("resilience")
    def check(self, key: str) -> bool:
        """Phase 0 stub — no real idempotency check yet."""
        return False

    @log_call("resilience")
    def record(self, key: str) -> None:
        """Phase 0 stub — no real recording yet."""
        pass
