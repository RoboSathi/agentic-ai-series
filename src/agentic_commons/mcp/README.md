# mcp — Model Context Protocol (parked)

Interface-only. The contracts for tools that arrive over MCP rather than
being written as Python functions. Whatever an MCP server offers must enter
as an ordinary `ToolDefinition` through the same registry, so the
Gatekeeper sees one kind of proposal, never a side door.

## Vocabulary

- **McpToolSource** — a tool source backed by an MCP session
- **ToolGovernancePolicy** — which remote tools may be exposed at all

## Public interface

- `McpServerSpec`, `McpSession`, `McpToolInfo`, `McpToolSource`, `McpTransport`, `ToolGovernancePolicy`, `TransportKind` (contracts only)

## Configuration

No config surface until an implementation lands.

## Seams

- Would plug into `tools.NamespacedToolRegistry` as a `ToolSource`.

## Collaborators

- `tools`, `gatekeeper`.

## Episodes

Planned with the tools episodes.

## Tests

Run with `./venv/bin/pytest tests/unit/test_platform_boundary.py -q` (or `make test` for everything).

## Status and limits

**Parked.** No transport is implemented and nothing calls these contracts. The boundary test only checks the package imports.
