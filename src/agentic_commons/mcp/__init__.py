"""Model Context Protocol — PARKED (LLD §16.1).

Interface-only this phase; the contracts below are normative. No code in
this repo calls them yet, and no transport is implemented.

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

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.tools.types import ToolDefinition

TransportKind = Literal["stdio", "http"]


class McpServerSpec(BaseModel):
    """How to reach one MCP server, and under whose name."""

    name: str
    transport: TransportKind = "stdio"
    # stdio: the process to launch. http: the base URL to call.
    command: list[str] = Field(default_factory=list)
    url: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    # Tools from this server are registered under this namespace, so two
    # servers offering `search` do not collide and an audit entry says
    # which one ran.
    namespace: str = ""


class McpToolInfo(BaseModel):
    """A tool as the server described it — a claim, not a contract."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ToolGovernancePolicy(Protocol):
    """Decides the execution envelope a remote tool is granted locally.

    Never delegated to the server. The server says what its tool does;
    this says what this deployment will let it do.
    """

    def envelope_for(self, server: McpServerSpec, tool: McpToolInfo) -> dict[str, Any]:
        """Governance fields to apply to this tool's `ToolDefinition`."""
        ...


@runtime_checkable
class McpSession(Protocol):
    """An open conversation with one MCP server."""

    async def list_tools(self) -> list[McpToolInfo]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

    async def close(self) -> None: ...


@runtime_checkable
class McpTransport(Protocol):
    """Opens a session to a server over one wire protocol."""

    async def connect(self, server: McpServerSpec) -> McpSession: ...


@runtime_checkable
class McpToolSource(Protocol):
    """An MCP server, presented as an ordinary source of governed tools.

    Structurally a `tools.ToolSource`: whatever it yields is a plain
    `ToolDefinition` and goes through the registry, the selector and the
    Gatekeeper exactly like a local tool. There is no second path.
    """

    async def tools(self) -> list[ToolDefinition]: ...


__all__ = [
    "McpServerSpec",
    "McpSession",
    "McpToolInfo",
    "McpToolSource",
    "McpTransport",
    "ToolGovernancePolicy",
    "TransportKind",
]
