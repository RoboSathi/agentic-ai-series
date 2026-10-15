"""Guardrail value types (LLD §9)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# reject: halt on invalid output. auto_repair: try to fix it, bounded.
# constrained: make invalid output unrepresentable at decode time.
GuardrailMode = Literal["reject", "auto_repair", "constrained"]


class ValidationResult(BaseModel):
    """Whether raw model output satisfied its schema, and what it cost."""

    valid: bool
    value: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    repaired: bool = False
    attempts: int = 1
    mode: GuardrailMode = "reject"
    raw: str = ""


class GuardrailViolationError(Exception):
    """Raised when output cannot be made valid and the mode forbids passing it.

    Loud by design. Silently coercing a malformed tool call into
    something plausible is how an agent acts on a value no model ever
    produced.
    """
