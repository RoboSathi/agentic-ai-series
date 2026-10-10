# agent_platform — the shared harness platform

The technical platform every agent in this repo composes: model adapters,
the harness controls (screening, gatekeeper, grounding, memory, context),
the always-on plumbing (resilience, durability, observability, audit,
logging, persistence), the operator console, and the eval framework.

It names no agent. An agent reaches it by injection and configuration —
`AgentHost` for the console, `EvalPlugin` for evals, `AGENTIC_OVERLAYS` for
its config layer — and `tests/unit/test_platform_boundary.py` fails if any
module, config file or document here names one.

## Modules

| Module | Owns |
|---|---|
| [audit](audit/README.md) | Append-only decision trail |
| [deterministic_checks](deterministic_checks/README.md) | Pre-LLM check contract (the business-logic seam) |
| [commands](commands/README.md) | Operator slash commands |
| [config](config/README.md) | Layered YAML, typed settings, the feature tree |
| [context](context/README.md) | Context assembly, strategies, failure detection |
| [loop_control](loop_control/README.md) | Loop watchdogs and contracts |
| [durability](durability/README.md) | Checkpointed, resumable runs; HITL interrupts |
| [evals](evals/README.md) | NOLLM cascade, eval pyramid, and the eval pipeline |
| [executor](executor/README.md) | Running approved tool calls |
| [gatekeeper](gatekeeper/README.md) | Deterministic authorization of every action |
| [guardrails](guardrails/README.md) | Structured output and the response guardrail |
| [hallucination_protection](hallucination_protection/README.md) | Claim-level grounding |
| [intent](intent/README.md) | Intent-classification contracts and provider |
| [llm](llm/README.md) | Model adapters, rate-limited HTTP, `make llm-check` |
| [logging](logging/README.md) | File/console logs and traced calls |
| [mcp](mcp/README.md) | Model Context Protocol (parked) |
| [memory](memory/README.md) | The four CoALA stores, governance, RAG |
| [multi_agent](multi_agent/README.md) | Not yet designed |
| [observability](observability/README.md) | Feed, traces, cost, SLOs, analytics store |
| [persistence](persistence/README.md) | SQLite access and platform tables |
| [prompt_compilation](prompt_compilation/README.md) | Compiled prompts (parked) |
| [resilience](resilience/README.md) | Nygard patterns and the cross-process rate limiter |
| [sandbox](sandbox/README.md) | Execution isolation (parked) |
| [security](security/README.md) | RBAC, PII, prompt injection, moderation, hosted guards |
| [tools](tools/README.md) | Typed tools and retrieval-based selection |
| [tui](tui/README.md) | The operator console |
| [verification](verification/README.md) | Reading side effects back |

## Rules that hold everywhere

- Every value comes from `config/config.yaml` (and overlays, `local.yaml`, `SECTION__FIELD` env vars); nothing is hard-coded.
- Features are read once, in an agent's composition root; a withheld feature is a stage that is never built.
- Every hosted model request draws on the shared rate-limit budget (`resilience.rate_limit`).
- Verification is `make verify`: ruff, mypy `--strict`, pytest with the 80% coverage gate.
