"""What an MCP client and an MCP server built here share: the protocol's vocabulary and stdio."""

from agent_platform.knowledge.mcp.common.stdio import silence_console
from agent_platform.knowledge.mcp.common.types import (
    McpCallResult,
    McpServerInfo,
    McpToolInfo,
    TransportKind,
)

__all__ = ["McpCallResult", "McpServerInfo", "McpToolInfo", "TransportKind", "silence_console"]
