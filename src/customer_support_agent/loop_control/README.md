# loop_control — the output quality gate

The tool-use loop itself is the platform's `LoopGovernor`. This package is the output quality gate: a plan looked up per intent, one deterministic critique, at most one repair, an independent score.

- **Entry points:** `OutputQualityGate`, `PlanLookup`, `GroundedCritic`, `IndependentScorer` (`quality_gate.py`)
- **Configuration:** `customer_support_agent.plans`, `max_tool_iterations`
- **Design:** [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent)
