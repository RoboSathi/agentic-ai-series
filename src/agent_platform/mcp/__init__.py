"""Model Context Protocol: remote tools, entering as ordinary governed tools (LLD §16.1).

The contracts (`types`) are implemented over the MCP SDK: `SdkMcpSession` (one
open session, owned by its own task), `SdkMcpTransport` (stdio or HTTP, from an
agent's `mcp.servers`) and `GovernedMcpToolSource` (the server's tools as
`ToolDefinition`s under a locally configured envelope).

The idea: a tool need not be a Python function the agent's author wrote.
An MCP server publishes tools, resources and prompts over a transport,
and an agent that speaks the protocol can use them without its author
having written any of them.

Which is why this package is small, and why it lives here rather than
inside `multi_agent`. MCP is a *tool-transport* concern: whatever an MCP
server offers has to arrive as an ordinary `ToolDefinition` and enter
through the same `NamespacedToolRegistry` as everything else, so that the
Gatekeeper sees one kind of proposal and not two. A remote tool that
reached the executor by a private side door would be a tool nobody
authorized. Hence `McpToolSource` below is just a `tools.ToolSource` —
the seam already exists; this package only names what fills it.

The governance consequence is the part worth designing before any code:
a `ToolDefinition` carries `surface`, `reversible`, `side_effecting` and
`requires_authority_rank`, and a *remote* server is not a trustworthy
source for any of them. A server that describes its own tool as
reversible has asserted, not established, that.
So `McpToolSource` takes a `ToolGovernancePolicy`: the envelope is
supplied locally, per server, by the operator who decided to trust that
server at all. The description is the server's; the authority is not.
This is the same rule as the Authority Hierarchy — `tool_observations`
rank below `user_task`, and a tool catalogue is a tool observation.
"""

from agent_platform.mcp.session import McpSessionClosedError, SdkMcpSession
from agent_platform.mcp.source import ConfiguredGovernancePolicy, GovernedMcpToolSource
from agent_platform.mcp.transport import SdkMcpTransport, server_spec
from agent_platform.mcp.types import (
    McpCallResult,
    McpServerSpec,
    McpSession,
    McpToolInfo,
    McpToolSource,
    McpTransport,
    ToolGovernancePolicy,
    TransportKind,
)

__all__ = [
    "ConfiguredGovernancePolicy",
    "GovernedMcpToolSource",
    "McpCallResult",
    "McpServerSpec",
    "McpSession",
    "McpSessionClosedError",
    "McpToolInfo",
    "McpToolSource",
    "McpTransport",
    "SdkMcpSession",
    "SdkMcpTransport",
    "ToolGovernancePolicy",
    "TransportKind",
    "server_spec",
]
