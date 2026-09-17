"""Type definitions for circuit breaker."""

from typing import Any

from pydantic import BaseModel


class CircuitBreakerResult(BaseModel):
    """Result from circuit breaker."""

    success: bool = False
    result: Any = None

    model_config = {"arbitrary_types_allowed": True}
