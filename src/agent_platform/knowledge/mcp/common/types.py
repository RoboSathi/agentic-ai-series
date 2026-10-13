"""What both sides of the wire speak: a transport, a server's handshake, a tool, a call's result.

The client reads these off the protocol (`mcp.client`); a server built here
produces them (`mcp.server`). Neither side's policy lives here.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

TransportKind = Literal["stdio", "http"]


class McpToolInfo(BaseModel):
    """A tool as the server described it — a claim, not a contract."""

    name: str
    title: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    # What a structured result holds (empty: the tool answers in text only).
    output_schema: dict[str, Any] = Field(default_factory=dict)
    # The server's hints about the tool (readOnlyHint, destructiveHint, idempotentHint,
    # openWorldHint): claims to show beside the local governance, never to trust in its place.
    annotations: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class McpServerInfo(BaseModel):
    """What the server said about itself in the handshake."""

    name: str = ""
    version: str = ""
    protocol_version: str = ""
    instructions: str = ""
    # The capabilities it declared (tools, resources, prompts, logging, ...), by name and in
    # full (each with its flags, e.g. tools.listChanged); and what this client declared.
    capabilities: list[str] = Field(default_factory=list)
    server_capabilities: dict[str, Any] = Field(default_factory=dict)
    client_capabilities: dict[str, Any] = Field(default_factory=dict)
    # How the version was negotiated (`initialize` or `server/discover`), who asked, how long.
    handshake: str = ""
    client: str = ""
    duration_ms: float = 0.0


class McpCallResult(BaseModel):
    """What a remote tool returned: its text, its structured content, and whether it failed."""

    is_error: bool = False
    text: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    # The round trip as the client measured it.
    duration_ms: float = 0.0


__all__ = ["McpCallResult", "McpServerInfo", "McpToolInfo", "TransportKind"]
