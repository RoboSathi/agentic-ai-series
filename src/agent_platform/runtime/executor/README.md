# runtime/executor — the deterministic execution boundary

Runs a tool call the Gatekeeper approved, and nothing else: deadline, idempotency, lifecycle events and an audit hook. Always on — with the Gatekeeper withheld, a call runs as UNGOVERNED and is still audited.

- **Entry points:** `ToolExecutor`, `ToolExecutorImpl`, `current_execution_key`, `RegistryToolHandler`
- **Configuration:** none (timeouts from `resilience.timeout`)
- **Design:** [LLD §A9](../../../../design_docs/LLD.md#a9-runtime-executor-verification-scheduling) · [ADR P-10](../../../../design_docs/ADR.md#p-10--prd-g1-is-retired-the-gatekeeper-may-be-withheld-the-executor-may-not)
