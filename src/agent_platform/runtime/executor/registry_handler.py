"""The Executor's handler for tools that carry their own implementation.

A tool from a `ToolSource` (an MCP server, a generated client) arrives with its
handler attached to its `ToolDefinition`. This adapter lets the Executor run it
like any agent-owned handler: the approved call names the tool, the registry
finds it, and the tool's own handler runs with the validated arguments.
"""

import inspect
from typing import Any

from agent_platform.tools.registry import ToolRegistry
from agent_platform.tools.types import ToolCall, ToolResult


class UnknownToolError(LookupError):
    """An approved call named a tool the registry does not hold."""


class RegistryToolHandler:
    """A `ToolExecutionHandler` over a registry of self-implementing tools (Adapter)."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._registry.get(call.action)
        if tool is None:
            raise UnknownToolError(call.action)
        outcome: Any = tool.handler(**call.params)
        result = await outcome if inspect.isawaitable(outcome) else outcome
        return (
            result
            if isinstance(result, ToolResult)
            else ToolResult(success=True, output=str(result))
        )


__all__ = ["RegistryToolHandler", "UnknownToolError"]
