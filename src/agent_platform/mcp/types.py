"""The MCP contracts: how a server is reached, what it offers, and who governs it.

Normative since LLD §16.1; implemented by `session`, `transport` and `source`.
"""

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_platform.tools.types import ToolDefinition

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


class McpServerInfo(BaseModel):
    """What the server said about itself in the handshake."""

    name: str = ""
    version: str = ""
    protocol_version: str = ""
    instructions: str = ""
    # The capabilities it declared (tools, resources, prompts, logging, ...).
    capabilities: list[str] = Field(default_factory=list)


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


class McpCallResult(BaseModel):
    """What a remote tool returned: its text, its structured content, and whether it failed."""

    is_error: bool = False
    text: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)


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
    "McpCallResult",
    "McpServerDescription",
    "McpServerInfo",
    "McpToolDescription",
    "McpServerSpec",
    "McpSession",
    "McpToolInfo",
    "McpToolSource",
    "McpTransport",
    "ToolGovernancePolicy",
    "TransportKind",
]
