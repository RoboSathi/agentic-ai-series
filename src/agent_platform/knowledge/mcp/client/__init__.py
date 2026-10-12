"""The governed MCP client: a server's tools, entering as ordinary `ToolDefinition`s.

`SdkMcpTransport` opens an `SdkMcpSession` (one live session, owned by its own
task) from an agent's `mcp.servers`; `GovernedMcpToolSource` offers the server's
tools under the envelope this deployment grants (`ConfiguredGovernancePolicy`).
"""

from agent_platform.knowledge.mcp.client.contracts import (
    McpServerDescription,
    McpServerSpec,
    McpSession,
    McpToolDescription,
    McpToolSource,
    McpTransport,
    ToolGovernancePolicy,
)
from agent_platform.knowledge.mcp.client.governed_source import (
    ConfiguredGovernancePolicy,
    GovernedMcpToolSource,
)
from agent_platform.knowledge.mcp.client.sdk_session import McpSessionClosedError, SdkMcpSession
from agent_platform.knowledge.mcp.client.sdk_transport import SdkMcpTransport, server_spec

__all__ = [
    "ConfiguredGovernancePolicy",
    "GovernedMcpToolSource",
    "McpServerDescription",
    "McpServerSpec",
    "McpSession",
    "McpSessionClosedError",
    "McpToolDescription",
    "McpToolSource",
    "McpTransport",
    "SdkMcpSession",
    "SdkMcpTransport",
    "ToolGovernancePolicy",
    "server_spec",
]
