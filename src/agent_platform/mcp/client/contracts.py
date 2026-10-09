"""The MCP client's contracts: how a server is reached, what it offers, and who governs it.

Normative since LLD §16.1; implemented by `sdk_session`, `sdk_transport` and
`governed_source`. What both sides of the wire speak is in `mcp.common.types`.
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_platform.mcp.common.types import (
    McpCallResult,
    McpServerInfo,
    McpToolInfo,
    TransportKind,
)
from agent_platform.tools.types import ToolDefinition


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


@runtime_checkable
class ToolGovernancePolicy(Protocol):
    """Decides the execution envelope a remote tool is granted locally.

    Never delegated to the server. The server says what its tool does;
    this says what this deployment will let it do.
    """

    def envelope_for(self, server: McpServerSpec, tool: McpToolInfo) -> dict[str, Any]:
        """Governance fields to apply to this tool's `ToolDefinition`."""
        ...


class McpToolDescription(BaseModel):
    """One tool as an agent is offered it: the server's description, the local governance."""

    name: str
    qualified_name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    envelope: dict[str, Any] = Field(default_factory=dict)


class McpServerDescription(BaseModel):
    """A connected server, everything an operator needs to understand what it offers."""

    server: McpServerSpec
    info: McpServerInfo
    tools: list[McpToolDescription] = Field(default_factory=list)


@runtime_checkable
class McpSession(Protocol):
    """An open conversation with one MCP server."""

    def server_info(self) -> McpServerInfo: ...

    async def list_tools(self) -> list[McpToolInfo]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult: ...

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

    def describe(self) -> McpServerDescription | None:
        """The connected server and the tools it offered (None before it connected)."""
        ...


__all__ = [
    "McpServerDescription",
    "McpServerSpec",
    "McpSession",
    "McpToolDescription",
    "McpToolSource",
    "McpTransport",
    "ToolGovernancePolicy",
]
