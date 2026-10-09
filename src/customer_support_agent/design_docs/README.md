# Customer Support Agent — design notes

The series' recurring running project: a customer-support refund agent
that **composes** `agent_platform` rather than duplicating it. This page
records what can be verified from the code as it stands; it adds no
decisions after the fact. The series' original PRD/HLD/LLD live in the
top-level [`design_docs/`](../../../design_docs/README.md).

## Running it

```
make dummy-data csa       # rebuild the agent database (KEEP_MEMORY=1 keeps memory)
make agent csa        # the operator console
make eval csa       # generate → oracle → security (Promptfoo) → judge → deepeval
```

Admin (csa only): `make refund-limit <amount>`, `make show-memory`,
`make dream`, `make clean-memory`.

## The graph

From `graph/agent.py` and `graph/edges.py`:

```
authenticate → choose_order → [pii_filtering → prompt_injection →
content_moderation → classify_intent → assemble_context] → propose ⇄ execute
→ finalize [→ await_approval]
```

- **Every turn is a durable run** — its own checkpointed thread
  (`DurableGraphRunner`); a refund above the limit pauses in
  `await_approval` (an `interrupt`) until the reviewer's decision resumes
  the same thread.
- **Authentication is the hard front door**: an unauthenticated request
  never reaches screening, intent, retrieval or the model.
- **Front-door stages are built only when their feature is on** and their
  collaborator exists; edges connect each to the next one that exists.
- **`propose ⇄ execute` is the Tool-Use Loop**, bounded by
  `max_tool_iterations`: the model proposes a typed tool call; the harness
  (Gatekeeper, executor) decides whether it happens. It is deliberately
  not ReAct.
- The model call goes through a timeout, a retry under an idempotency key
  and a circuit breaker (`graph/nodes/propose.py`).

## Where things are

| package | what it owns |
|---|---|
| `graph/` | the agent, the wiring (`edges.py`), node groups (`nodes/`), the composition root (`builder.py`) |
| `intent/router.py` | out-of-scope rejection → classifier → one clarification question |
| `basic_sanity/order_validator.py` | `OrderSanityCheck`, csa's business-logic seam into the platform's `BasicSanityChecker` |
| `gatekeeper/` | csa's business rules and mandates for the platform Gatekeeper |
| `loop_control/quality_gate.py` | the Output Quality Gate: `PlanLookup`, `GroundedCritic`, `IndependentScorer` |
| `resilience/` | the refund and cancellation sagas (compensating steps, notification last) |
| `review/` | the human review queue and referral |
| `domain/` | orders, identity, refund and cancellation policy |
| `persistence/` | the agent database schema and repositories |
| `tools/definitions.py` | the typed tools the model may propose |
| `verification/refund_effects.py` | read-back of a refund's side effects |
| `console/` | the host the shared TUI composes, and csa's commands |
| `evals/` | the eval plugin, fixture, oracle invariants, golden cases, Promptfoo security suite |
| `admin/` | memory inspection, consolidation (`dream`), refund limit |

## Features

csa's overlay turns on every platform feature (`basic_sanity`,
`prompt_injection`, `content_moderation`, `pii`, `intent_classification`,
`hallucination_guard`, `gatekeeper`, `memory` with all four CoALA stores,
`context` with rerank and failure detection). `/harness` moves them for
the running process only.

## Patterns

See [`patterns.md`](patterns.md).
