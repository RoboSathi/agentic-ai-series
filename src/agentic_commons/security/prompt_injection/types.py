"""Type definitions for prompt injection detection."""

from pydantic import BaseModel


class InjectionResult(BaseModel):
    """Result of injection detection."""

    is_injection: bool = False
    confidence: float = 0.0
