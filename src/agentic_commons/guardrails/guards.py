"""Guardrail stubs for Phase 0."""

from typing import Any

from agentic_commons.guardrails.types import ConstraintResult, ValidationOutcome
from agentic_commons.logging import log_call


class PydanticGuard:
    """Pydantic-based validation guard stub."""

    @log_call("guardrails")
    def validate(self, data: Any, schema: type) -> ValidationOutcome:
        """Phase 0 stub — no real validation yet."""
        return ValidationOutcome(valid=True, errors=[])


class OutlinesGuard:
    """Outlines grammar-based guard stub."""

    @log_call("guardrails")
    def constrain(self, prompt: str, grammar: str) -> ConstraintResult:
        """Phase 0 stub — no real constraint yet."""
        return ConstraintResult(output="", constrained=False)
