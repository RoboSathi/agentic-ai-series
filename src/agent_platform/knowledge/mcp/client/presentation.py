"""An MCP server as a card: what it is, what it says, and every tool it offers.

Any agent that consumes MCP draws this when it connects, so an operator sees the
protocol's side of the conversation: the server's own name, version and
instructions from the handshake, and each tool as the server describes it (its
arguments from the input schema), beside the governance this deployment grants
it — which is the part the server does not get to decide.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.knowledge.mcp.client.contracts import McpServerDescription, McpToolDescription
from agent_platform.knowledge.mcp.common.types import McpCallResult, McpServerInfo
from agent_platform.observability.step_events import EXCHANGE_PROTOCOL, MCP_PROTOCOL

# The protocol method a tool call is (the MCP CALL card names it).
TOOLS_CALL = "tools/call"

_PROPERTIES, _REQUIRED, _TYPE, _ANY_OF = "properties", "required", "type", "anyOf"
_UNTYPED = "any"


def server_card(description: McpServerDescription) -> tuple[str, dict[str, list[str]]]:
    """(heading, sections) of the MCP SERVER card: the session's protocol profile, once."""
    words = get_settings().mcp.card
    info, spec = description.info, description.server
    heading = words.heading.format(
        server=info.name or spec.name,
        protocol=info.protocol_version,
        tools=len(description.tools),
    )
    sections = {
        words.handshake_title: _handshake_lines(info),
        words.server_title: _server_lines(description),
        words.client_title: _client_lines(),
        words.capabilities_title: _capability_lines(info),
        words.instructions_title: [info.instructions or words.no_instructions],
        words.tools_title: [line for tool in description.tools for line in _tool_lines(tool)],
        words.execution_title: _execution_lines(),
    }
    return heading, sections


def _handshake_lines(info: McpServerInfo) -> list[str]:
    words, client = get_settings().mcp.card, get_settings().mcp.client
    values = {
        "method": info.handshake,
        "protocol": info.protocol_version,
        "mode": client.mode,
        "duration_ms": info.duration_ms,
    }
    return [line.format(**values) for line in words.handshake_lines]


def _server_lines(description: McpServerDescription) -> list[str]:
    info, spec = description.info, description.server
    values = {
        "name": info.name or spec.name,
        "version": info.version,
        "namespace": spec.namespace,
        "transport": spec.transport,
        "target": " ".join(spec.command) if spec.command else spec.url,
    }
    return [line.format(**values) for line in get_settings().mcp.card.server_lines]


def _client_lines() -> list[str]:
    client = get_settings().mcp.client
    values = {"name": client.name, "version": client.version, "log_level": client.log_level}
    return [line.format(**values) for line in get_settings().mcp.card.client_lines]


def _capability_lines(info: McpServerInfo) -> list[str]:
    """Each side's declared capabilities, with their flags (`tools: listChanged=false`)."""
    words = get_settings().mcp.card
    sides = (
        (info.server_capabilities, words.server_capability_line, words.server_side),
        (info.client_capabilities, words.client_capability_line, words.client_side),
    )
    lines: list[str] = []
    for declared, template, side in sides:
        if not declared:
            none = words.client_none if side == words.client_side else words.no_capabilities
            lines.append(none.format(side=side))
            continue
        lines.extend(
            template.format(name=name, flags=_flags(value)) for name, value in declared.items()
        )
    return lines


def _flags(value: Any) -> str:
    """A capability's sub-flags on one line, or that it is declared with none."""
    if isinstance(value, dict) and value:
        return " · ".join(f"{key}={_flags(item)}" for key, item in value.items())
    if isinstance(value, dict):
        return get_settings().mcp.card.no_flags
    return str(value).lower() if isinstance(value, bool) else str(value)


def _tool_lines(tool: McpToolDescription) -> list[str]:
    words = get_settings().mcp.card
    # A docstring's line breaks and indents are layout, not content: one line on the card.
    description = " ".join(tool.description.split())
    lines = [words.tool_line.format(name=tool.qualified_name, description=description)]
    if tool.offered_as:
        lines.append(words.offered_line.format(name=tool.offered_as))
    lines.extend(_input_lines(tool.input_schema))
    lines.extend(_output_lines(tool.output_schema))
    hints = [
        words.hint.format(name=name, value=_flags(value))
        for name, value in tool.annotations.items()
    ]
    lines.append(words.hints_line.format(hints=", ".join(hints) or words.no_hints))
    if tool.meta:
        lines.append(words.meta_line.format(meta=_flags(tool.meta)))
    envelope = tool.envelope
    lines.append(
        words.governance_line.format(
            surface=envelope.get("surface", ""),
            side_effecting=_flags(envelope.get("side_effecting", "")),
            reversible=_flags(envelope.get("reversible", "")),
            rank=envelope.get("requires_authority_rank", ""),
            permission=envelope.get("required_permission") or "",
        )
    )
    return lines


def _input_lines(schema: dict[str, Any]) -> list[str]:
    """Each argument of the inputSchema: its type, and whether it is required."""
    words = get_settings().mcp.card
    required = set(schema.get(_REQUIRED, []))
    return [
        words.argument_line.format(
            name=name,
            type=_type_of(field),
            required=words.required if name in required else words.optional,
        )
        for name, field in schema.get(_PROPERTIES, {}).items()
    ]


def _output_lines(schema: dict[str, Any]) -> list[str]:
    """Each field of the outputSchema, or that the tool answers in text only."""
    words = get_settings().mcp.card
    fields: dict[str, Any] = schema.get(_PROPERTIES, {})
    if not fields:
        return [words.no_output]
    return [words.output_line.format(name=name, type=_type_of(f)) for name, f in fields.items()]


def _execution_lines() -> list[str]:
    """How a call runs and fails: its deadline, how much of a result is kept, errors, _meta."""
    settings = get_settings().mcp
    words = settings.card
    limit = settings.result_text_limit
    values = {
        "timeout": settings.call_timeout_seconds,
        "limit": words.result_limit.format(limit=limit) if limit else words.whole_result,
        "meta_key": settings.client.log_level_meta_key,
        "log_level": settings.client.log_level,
    }
    return [line.format(**values) for line in words.execution_lines]


def _type_of(schema: dict[str, Any]) -> str:
    """An argument's JSON-schema type, `a | b` for a union (e.g. an optional integer)."""
    if _TYPE in schema:
        return str(schema[_TYPE])
    options = [str(option.get(_TYPE, _UNTYPED)) for option in schema.get(_ANY_OF, [])]
    return " | ".join(options) or _UNTYPED


def call_exchange(
    server: str, tool: str, arguments: dict[str, Any], result: McpCallResult
) -> dict[str, Any]:
    """One call as the protocol saw it, for the MCP CALL card (`ToolResult.exchange`)."""
    return {
        EXCHANGE_PROTOCOL: MCP_PROTOCOL,
        "server": server,
        "method": TOOLS_CALL,
        "tool": tool,
        "arguments": arguments,
        "is_error": result.is_error,
        "duration_ms": result.duration_ms,
        "structured_keys": sorted(result.structured),
    }


__all__ = ["TOOLS_CALL", "call_exchange", "server_card"]
