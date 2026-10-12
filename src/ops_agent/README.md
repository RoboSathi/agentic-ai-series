# ops_agent — analytics questions over MCP

Answers questions about how the agents are running — latency, tokens, backups, tool failures,
Gatekeeper decisions, SLOs — using only the platform's analytics MCP server, each call
validated, authorized, executed and audited like a local tool. Every figure in an answer must
appear in a query result; the reply shows its SQL and rows.

```
make analytics-export          # or keep any agent's console running
make agent ops                 # SOURCE=dummy reads the generated demo world
```

**Design:** [HLD §5.3](../../design_docs/HLD.md#53-ops-analyst) · [LLD §B3](../../design_docs/LLD.md#b3-ops-agent) · [ADR O-01](../../design_docs/ADR.md#o-01--the-ops-agents-only-tools-are-the-analytics-mcp-servers)
