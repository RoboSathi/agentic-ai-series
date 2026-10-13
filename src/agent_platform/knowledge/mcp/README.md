# knowledge/mcp — Model Context Protocol, client and servers

`common/` (protocol vocabulary), `client/` (a governed client: a server's tools become ordinary `ToolDefinition`s on the one tool path, under the deployment's envelope), `server/analytics/` (the read-only analytics server: `python -m agent_platform.knowledge.mcp.server.analytics`).

- **Entry points:** `SdkMcpTransport`, `SdkMcpSession`, `GovernedMcpToolSource`, `ConfiguredGovernancePolicy`, `server/analytics`
- **Configuration:** an agent's `mcp.servers[]`, `mcp.default_envelope`
- **Design:** [LLD §A23](../../../../design_docs/LLD.md#a23-knowledge-web-mcp) · [ADR P-25](../../../../design_docs/ADR.md#p-25--the-analytics-mcp-server-and-the-governed-mcp-client)
