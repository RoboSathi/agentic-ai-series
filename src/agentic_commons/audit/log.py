"""Audit log stubs for Phase 0."""

from agentic_commons.audit.types import AuditEvent
from agentic_commons.logging import log_call


class AuditLog:
    """Audit trail interface stub."""

    @log_call("audit")
    def record(self, event: AuditEvent) -> None:
        """Phase 0 stub — no real audit recording yet."""
        pass

    @log_call("audit")
    def query(self, filters: dict) -> list[AuditEvent]:
        """Phase 0 stub — no real query yet."""
        return []
