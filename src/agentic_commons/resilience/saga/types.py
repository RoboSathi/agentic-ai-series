"""Type definitions for saga pattern."""

from pydantic import BaseModel


class SagaStep(BaseModel):
    """A step in a saga."""

    name: str = ""


class SagaResult(BaseModel):
    """Result of saga execution."""

    completed: bool = False
    steps_executed: int = 0
