"""An MCP server as a card: what it is, what it says, and every tool it offers.

Any agent that consumes MCP draws this when it connects, so an operator sees the
protocol's side of the conversation: the server's own name, version and
instructions from the handshake, and each tool as the server describes it (its
arguments from the input schema), beside the governance this deployment grants
it — which is the part the server does not get to decide.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.mcp.types import McpServerDescription, McpToolDescription

_PROPERTIES, _REQUIRED, _TYPE, _ANY_OF = "properties", "required", "type", "anyOf"
_UNTYPED = "any"


def server_card(description: McpServerDescription) -> tuple[str, dict[str, list[str]]]:
    """(heading, sections) of the MCP SERVER card for a connected server."""
    words = get_settings().mcp.card
    info, spec = description.info, description.server
    heading = words.heading.format(
        server=info.name or spec.name,
        protocol=info.protocol_version,
        tools=len(description.tools),
    )
    target = " ".join(spec.command) if spec.command else spec.url
    sections = {
        words.server_title: [
            words.server_line.format(
                name=" ".join(part for part in (info.name or spec.name, info.version) if part),
                namespace=spec.namespace,
            ),
            words.transport_line.format(transport=spec.transport, target=target),
        ],
        words.instructions_title: [info.instructions or words.no_instructions],
        words.tools_title: [line for tool in description.tools for line in _tool_lines(tool)],
    }
    return heading, sections


def _tool_lines(tool: McpToolDescription) -> list[str]:
    words = get_settings().mcp.card
    lines = [words.tool_line.format(name=tool.qualified_name, description=tool.description)]
    properties: dict[str, Any] = tool.input_schema.get(_PROPERTIES, {})
    required = set(tool.input_schema.get(_REQUIRED, []))
    lines.extend(
        words.argument_line.format(
            name=name,
            type=_type_of(schema),
            required=words.required if name in required else words.optional,
        )
        for name, schema in properties.items()
    )
    envelope = tool.envelope
    lines.append(
        words.governance_line.format(
            surface=envelope.get("surface", ""),
            side_effecting=envelope.get("side_effecting", ""),
            permission=envelope.get("required_permission") or "",
        )
    )
    return lines


def _type_of(schema: dict[str, Any]) -> str:
    """An argument's JSON-schema type, `a | b` for a union (e.g. an optional integer)."""
    if _TYPE in schema:
        return str(schema[_TYPE])
    options = [str(option.get(_TYPE, _UNTYPED)) for option in schema.get(_ANY_OF, [])]
    return " | ".join(options) or _UNTYPED


__all__ = ["server_card"]
