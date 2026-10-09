# Generated analytics source (demo)

`make dummy-data analytics` writes the `dummy` analytics source here (gitignored,
regenerated on demand):

- `observability.duckdb`: the same tables a live agent writes (`fact_*`, the audit
  ledger, each agent's own Tier-2 facts). `agent_platform/analytics/dummy.py` fills them
  from every agent's profile (`src/<package>/analytics/dummy.yaml`) over
  `analytics.dummy.days`, with one incident day (`analytics.dummy.incident`).
- `parquet/`: its exported views (Parquet plus manifest), exactly as `make analytics-export`
  writes the live source's.

Use it with `SOURCE=dummy`:

```
make dummy-data analytics
make dashboard SOURCE=dummy      # Rill over the demo source
make agent ops SOURCE=dummy      # the MCP demo: ask the Ops Analyst about it (its own turns are recorded to the live source)
```

An external MCP client (Claude Code, MCP Inspector) can launch the server itself:
`ANALYTICS__SOURCE=dummy venv/bin/python -m agent_platform.mcp.server.analytics`.
The live source is never touched.
