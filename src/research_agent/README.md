# research_agent — scoped, audited web research

Answers a plain question directly (a ReAct loop over web search, page fetch, weather and the
clock, grounded against what the tools returned), and runs `/research` as a scoped interview,
a frozen and hashed contract, an autonomous DAG of research workers, an independent audit, and
a COMPLETE or PARTIAL report.

```
make agent ra           # the console (needs nvidia_api_key; tavily_api_key optional)
make dream ra           # consolidate what runs learned
```

**Design:** [HLD §5.2](../../design_docs/HLD.md#52-research) · [LLD §B2](../../design_docs/LLD.md#b2-research-agent) · [ADR §9](../../design_docs/ADR.md#9-research-agent)
