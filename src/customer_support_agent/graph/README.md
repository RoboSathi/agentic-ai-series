# graph — the LangGraph application and its composition root

The agent: its state, nodes, wiring and `build_agent`, which reads the feature tree once and builds only the declared stages. Every turn is a durable thread; a refund queued for a person pauses in `await_approval`.

- **Entry points:** `build_agent` (`builder.py`), `CustomerSupportAgent` (`agent.py`), `GraphWiring` (`edges.py`), `nodes/`
- **Configuration:** none (reads every section it wires)
- **Design:** [HLD §5.1](../../../design_docs/HLD.md#51-customer-support) · [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent)
