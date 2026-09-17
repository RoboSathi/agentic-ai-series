"""Hallucination checker stub for Phase 0."""

from agentic_commons.hallucination_protection.types import HallucinationVerdict
from agentic_commons.logging import log_call


class HallucinationChecker:
    """Hallucination detection interface stub."""

    @log_call("hallucination_protection")
    def check(self, claim: str, context: str) -> HallucinationVerdict:
        """Phase 0 stub — no real checking yet."""
        return HallucinationVerdict(is_hallucination=False, confidence=0.0)
