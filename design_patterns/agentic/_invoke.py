"""Run one validated tool call, for the teaching patterns.

The platform no longer lets a registry execute what it holds: execution
sits behind the executor, where idempotency, deadlines and the
Gatekeeper live. The patterns in this package are shown *without* that
boundary on purpose — they teach the loop shape, not governance — so
they call the tool's own handler here, and turn any escape into a failed
result the model can see and recover from.
"""

import inspect

from agent_platform.logging import get_logger
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.types import ToolCall, ToolResult


async def invoke_tool(registry: NamespacedToolRegistry, call: ToolCall) -> ToolResult:
    """Execute a validated call, wrapping any exception into a failed result."""
    tool = registry.get(call.action)
    if tool is None:
        return ToolResult(success=False, error=f"Unknown tool: {call.action}")
    try:
        outcome = tool.handler(**call.params)
        result = await outcome if inspect.isawaitable(outcome) else outcome
    except Exception as exc:
        get_logger("design_patterns").exception("Tool raised", tool=call.action)
        return ToolResult(success=False, error=f"Tool '{call.action}' failed: {exc}")
    return (
        result if isinstance(result, ToolResult) else ToolResult(success=True, output=str(result))
    )
