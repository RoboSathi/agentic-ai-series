"""Type definitions for tools module."""

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result of tool invocation."""

    success: bool = False
    output: str = ""
