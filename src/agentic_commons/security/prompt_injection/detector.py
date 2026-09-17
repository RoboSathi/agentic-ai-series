"""Prompt injection detection stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.security.prompt_injection.types import InjectionResult


class InjectionDetector:
    """Prompt injection detection interface stub."""

    @log_call("security", toggle_path="security.prompt_injection.enabled")
    def check(self, text: str) -> InjectionResult:
        """Phase 0 stub — no real detection yet."""
        return InjectionResult(is_injection=False, confidence=0.0)
