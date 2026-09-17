"""Tool registry stub for Phase 0."""

from typing import Any

from agentic_commons.logging import log_call
from agentic_commons.tools.types import ToolResult


class ToolRegistry:
    """Tool registry interface stub."""

    @log_call("tools")
    def register(self, tool: Any) -> None:
        """Phase 0 stub — no real registration yet."""
        pass

    @log_call("tools")
    def invoke(self, name: str, args: dict) -> ToolResult:
        """Phase 0 stub — no real invocation yet."""
        return ToolResult(success=False, output="")
