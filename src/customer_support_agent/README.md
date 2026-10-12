# customer_support_agent — the refund agent

A customer-support agent that answers order and refund questions, issues refunds within its
authority, queues larger ones for a person, and refuses what policy forbids. It composes
`agent_platform` with every harness feature declared; business rules live in code
(`domain/`), never only in the prompt.

```
make dummy-data csa     # build its database from the seed
make agent csa          # the console
make eval csa           # oracle, DeepEval, Promptfoo over the core set
```

**Design:** [HLD §5.1](../../design_docs/HLD.md#51-customer-support) · [LLD §B1](../../design_docs/LLD.md#b1-customer-support-agent) · [ADR §10](../../design_docs/ADR.md#10-customer-support-agent)
