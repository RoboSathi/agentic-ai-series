# Ops agent — patterns

Where this agent uses each design pattern, derived from the code as it stands.

## Workflow and agentic patterns

| Pattern | Where | Notes |
|---|---|---|
| react | graph/graph.py · reason ⇄ act | the model explores the source (list → describe → query) and corrects its SQL from the server's refusals, bounded by `ops_agent.max_steps` |
| procedural memory | prompts/AGENT.md + prompts/skills/analytics-investigation · graph/agent.py `_messages` | the profile and its one `always` procedure, versioned in `SkillStore` and audited per turn (ADR 0010) |
| tool_use (MCP) | graph/builder.py · `_sources` → `GovernedMcpToolSource` | the only tools are a remote MCP server's, arriving as ordinary governed `ToolDefinition`s |
| guardrails | graph/agent.py · `_screened` | the person's message is screened for injection before the model reads it (`prompt_injection`) |
| hallucination_check | graph/nodes.py · `_check` → platform `ConfidenceCheck` (level 1), then guardrails/figures.py · `FigureCheck` (level 2) | level 1: token confidence (logprobs), shown first on HALLUCINATION CHECK as a signal; level 2: every figure must appear in a query result, the rest reported as unverified (`hallucination_guard`) |
| human-verifiable output | presentation/reply.py | the reply carries the SQL it ran and the rows it rests on |

## Governance and LLD patterns

| Pattern | Where |
|---|---|
| gatekeeper (authority only) | graph/tool_calls.py · `GovernedCalls` — VALID when allowed, VOID refused, UNGOVERNED when withheld |
| adapter | agent_platform/mcp/session.py (SDK client), executor/registry_handler.py |
| active object | agent_platform/mcp/session.py · one task owns the MCP session |
| strategy | agent_platform/analytics/query_guard.py |
| facade | agent_platform/analytics/source_db.py · `SourceDatabase` |
| factory / composition root | graph/builder.py — features are read once, here |
