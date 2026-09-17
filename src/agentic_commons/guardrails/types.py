"""Type definitions for guardrails module."""

from pydantic import BaseModel


class ValidationOutcome(BaseModel):
    """Outcome of validation."""

    valid: bool = True
    errors: list[str] = []


class ConstraintResult(BaseModel):
    """Result of constraint application."""

    output: str = ""
    constrained: bool = False
