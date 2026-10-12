# agent_platform — the shared harness platform

The platform every agent in this repository composes: model access, the harness controls
(screening, deterministic checks, intent, Gatekeeper, grounding, memory, context), the
always-on plumbing (resilience, durability, observability, audit, review, logging,
persistence), the operator console, analytics and evals. It names no agent: an agent reaches
it through `AgentHost`, `EvalPlugin` and its config overlay.

| Group | Packages |
|---|---|
| top level | `config` `logging` `llm` `judge` `scoring` `tools` `loop_control` `gatekeeper` `durability` `security` `deterministic_checks` `intent` `context` `memory` `hallucination_check` `guardrails` `resilience` `observability` `audit` `review` `evals` `analytics` |
| `core/` | `primitives` `persistence` |
| `interface/` | `tui` `commands` `presentation` `host` |
| `runtime/` | `executor` `scheduling` `verification` |
| `knowledge/` | `web` `mcp` `artifacts` |

**Design:** [HLD §3](../../design_docs/HLD.md#3-platform-component-map) · [LLD Part A](../../design_docs/LLD.md#part-a--platform) · [ADR](../../design_docs/ADR.md)
