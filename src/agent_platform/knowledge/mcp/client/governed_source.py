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
from agent_platform.knowledge.mcp.client.events import LoggedMcpEvents, McpEventSink
from agent_platform.knowledge.mcp.client.presentation import call_exchange
from agent_platform.knowledge.mcp.client.schema import arguments_model
from agent_platform.knowledge.mcp.common.types import McpToolInfo
from agent_platform.tools.types import ToolDefinition, ToolResult


class ConfiguredGovernancePolicy:
    """The envelope each server's tools run under: a tool's own (`mcp.servers[].tools`), else
    its server's (`mcp.servers[].envelope`), else the most restrictive default.

    A per-tool envelope narrows one tool without the others, e.g. free SQL needing a
    permission the catalogue tools do not, so the Gatekeeper can refuse it on its own.
    """

    def __init__(self, servers: list[McpServerSettings] | None = None) -> None:
        settings = get_settings().mcp
        configured = servers or settings.servers
        self._default = settings.default_envelope
        self._envelopes = {server.name: server.envelope for server in configured}
        self._tools = {server.name: server.tools for server in configured}

    def envelope_for(self, server: McpServerSpec, tool: McpToolInfo) -> dict[str, Any]:
        own = self._tools.get(server.name, {}).get(tool.name)
        envelope = own or self._envelopes.get(server.name) or self._default
        return envelope.model_dump()


class GovernedMcpToolSource:
    """One server's tools, connected on first use and governed locally."""

    def __init__(
        self,
        server: McpServerSpec,
        transport: McpTransport,
        policy: ToolGovernancePolicy,
        events: McpEventSink | None = None,
    ) -> None:
        self._app_config = get_settings()
        self._server = server
        self._transport = transport
        self._policy = policy
        # Every protocol event goes here: the log always, the live feed when an agent binds one.
        self._events = events or LoggedMcpEvents()
        self._session: McpSession | None = None
        self._listed: list[McpToolInfo] = []
        self._offered: dict[str, str] = {}

    async def tools(self) -> list[ToolDefinition]:
        session = await self._open()
        self._listed = await session.list_tools()
        definitions = [self._definition(tool) for tool in self._listed]
        # The name each tool is offered to the model by; the session profile logs it once.
        self._offered = {definition.name: definition.model_name for definition in definitions}
        description = self.describe()
        if description is not None:
            self._events.tools_listed(description)
        return definitions

    def stale(self) -> bool:
        """Whether the server changed its tools since they were listed."""
        return self._session is not None and self._session.tools_changed()

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
                    offered_as=self._offered.get(tool.name, ""),
                    title=tool.title,
                    output_schema=tool.output_schema,
                    annotations=tool.annotations,
                    meta=tool.meta,
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

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _open(self) -> McpSession:
        if self._session is None:
            try:
                self._session = await self._transport.connect(self._server, self._events)
            except Exception as exc:
                self._events.connect_failed(self._server.name, str(exc) or repr(exc))
                raise
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
            limit = self._app_config.mcp.result_text_limit
            text = result.text[:limit] if limit else result.text
            exchange = call_exchange(self._server.name, name, given, result)
            if result.is_error:
                return ToolResult(
                    success=False, error=text, data=result.structured, exchange=exchange
                )
            return ToolResult(success=True, output=text, data=result.structured, exchange=exchange)

        return call


__all__ = ["ConfiguredGovernancePolicy", "GovernedMcpToolSource"]
