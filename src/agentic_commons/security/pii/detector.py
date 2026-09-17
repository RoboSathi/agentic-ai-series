"""PII detection stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.security.pii.types import PIIScanResult


class PIIDetector:
    """PII detection interface stub."""

    @log_call("security", toggle_path="security.pii.enabled")
    def scan(self, text: str) -> PIIScanResult:
        """Phase 0 stub — no real detection yet."""
        return PIIScanResult(clean=True, findings=[])
