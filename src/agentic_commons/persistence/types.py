"""Type definitions for persistence module."""

from typing import Any

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Result of a database execution."""

    rows: list[dict[str, Any]] = []
    affected: int = 0
