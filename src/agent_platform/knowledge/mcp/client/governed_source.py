"""A remote server's tools, as governed `ToolDefinition`s (`McpToolSource`).

Each tool the server lists becomes a definition in the server's namespace, with
its arguments validated by a model built from the tool's schema, and executed
by calling the server. The governance envelope — surface, reversibility, side
effects, authority rank, permission — comes from this deployment's
configuration for that server (`ConfiguredGovernancePolicy`), never from the
server: a server describing its own tool as harmless has asserted it, not
established it.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import McpServerSettings
from agent_platform.knowledge.mcp.client.contracts import (
    McpServerDescription,
    McpServerSpec,
    McpSession,
    McpToolDescription,
    McpTransport,
    ToolGovernancePolicy,
)
from agent_platform.knowledge.mcp.client.presentation import call_exchange
from agent_platform.knowledge.mcp.client.schema import arguments_model
from agent_platform.knowledge.mcp.common.types import McpToolInfo
from agent_platform.logging import get_logger
from agent_platform.tools.types import ToolDefinition, ToolResult


class ConfiguredGovernancePolicy:
    """The envelope each server's tools run under, from `mcp.servers[].envelope`."""

    def __init__(self, servers: list[McpServerSettings] | None = None) -> None:
        settings = get_settings().mcp
        self._default = settings.default_envelope
        self._envelopes = {server.name: server.envelope for server in (servers or settings.servers)}

    def envelope_for(self, server: McpServerSpec, tool: McpToolInfo) -> dict[str, Any]:
        envelope = self._envelopes.get(server.name) or self._default
        return envelope.model_dump()


class GovernedMcpToolSource:
    """One server's tools, connected on first use and governed locally."""

    def __init__(
        self, server: McpServerSpec, transport: McpTransport, policy: ToolGovernancePolicy
    ) -> None:
        self._server = server
        self._transport = transport
        self._policy = policy
        self._session: McpSession | None = None
        self._listed: list[McpToolInfo] = []
        self._logger = get_logger("mcp")

    async def tools(self) -> list[ToolDefinition]:
        session = await self._open()
        self._listed = await session.list_tools()
        definitions = [self._definition(tool) for tool in self._listed]
        for definition in definitions:
            self._logger.info(
                "MCP tool registered",
                server=self._server.name,
                tool=definition.qualified_name,
                offered_as=definition.model_name,
                envelope=self._policy.envelope_for(self._server, self._tool(definition.name)),
            )
        return definitions

    def describe(self) -> McpServerDescription | None:
        """The connected server and the tools it offered (None before it connected)."""
        if self._session is None:
            return None
        return McpServerDescription(
            server=self._server,
            info=self._session.server_info(),
            tools=[
                McpToolDescription(
                    name=tool.name,
                    qualified_name=f"{self._server.namespace}.{tool.name}"
                    if self._server.namespace
                    else tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    envelope=self._policy.envelope_for(self._server, tool),
                )
                for tool in self._listed
            ],
        )

    def _tool(self, name: str) -> McpToolInfo:
        return next(tool for tool in self._listed if tool.name == name)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _open(self) -> McpSession:
        if self._session is None:
            self._session = await self._transport.connect(self._server)
        return self._session

    def _definition(self, tool: McpToolInfo) -> ToolDefinition:
        return ToolDefinition(
            name=tool.name,
            namespace=self._server.namespace,
            description=tool.description,
            args_schema=arguments_model(tool.name, tool.input_schema),
            handler=self._handler(tool.name),
            **self._policy.envelope_for(self._server, tool),
        )

    def _handler(self, name: str) -> Callable[..., Awaitable[ToolResult]]:
        async def call(**arguments: Any) -> ToolResult:
            session = await self._open()
            given = {key: value for key, value in arguments.items() if value is not None}
            result = await session.call_tool(name, given)
            limit = get_settings().mcp.result_text_limit
            text = result.text[:limit] if limit else result.text
            exchange = call_exchange(self._server.name, name, given, result)
            if result.is_error:
                return ToolResult(
                    success=False, error=text, data=result.structured, exchange=exchange
                )
            return ToolResult(success=True, output=text, data=result.structured, exchange=exchange)

        return call


__all__ = ["ConfiguredGovernancePolicy", "GovernedMcpToolSource"]
