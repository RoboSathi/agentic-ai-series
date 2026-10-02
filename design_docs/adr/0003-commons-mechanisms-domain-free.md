# ADR 0003 — Commons mechanisms for a second agent stay domain-free

## Context

The research agent needed capabilities the platform did not have. Each
could have been written inside ra, or written in commons in ra's words.
The platform's rule is that dependencies point agent → platform, and the
repo has regressed before by duplicating platform code inside an agent.

## Decision

After searching commons for an equivalent (the reuse map in
`src/research_agent/design_docs/BUILD_PLAN.md`), the following were added
as **domain-free mechanisms**, each with unit tests:

| module | mechanism |
|---|---|
| `scheduling/` | `TaskGraph` (validation, cycle finding, waves) + `AsyncDagScheduler` (bounded concurrency, failure isolation) |
| `control_loop/budget.py` | `BudgetGuard`: synchronous reserve → commit/release on named counters + wall clock |
| `artifacts/` | `FrozenArtifact` (canonical hash, verify) + `DirectoryArtifactStore` |
| `evidence/` | append-only `EvidenceLedger`, `VerbatimSpanValidator` / `NonEmptySpanValidator` |
| `web/` | `WebGateway` (live / record / replay), `SearchRouter` over `DuckDuckGoProvider` + `TavilyProvider` (time ranges, news vertical), on-disk `SearchCache`, HTTP fetcher with reader fallback, `FixtureStore` |
| `resilience/rate_limiter` (extended) | per-bucket **monthly quota** in the shared SQLite store; `QuotaExhaustedError` fails fast and is never retried |
| `llm/roles.py` | `RoleModelRouter` + distinct-model check |
| `durability/isolated.py`, `durability/hitl.py` | `IsolatedGraphRunner` (declared inputs only), `await_decision` |
| `verification/acceptance.py` | `AcceptanceCheckRegistry` of named checks by kind |
| `hashing.py` | `canonical_hash` moved from `evals.contracts` (re-exported there) |

Existing modules were **reused** wherever they fit: `DurableGraphRunner`,
`RunRegistry`, `InstructionDataSeparator`, `EntailmentModel`,
`PydanticValidator`, `ResilientCaller` and the rate limiter, the four
memory stores behind `MemoryFacade`, `AuditLedger`, `StageReporter`, the
eval pipeline and `FeatureSet`.

**The vocabulary guard.** `tests/agentic_commons/test_roles_and_boundary.py`
bans `archetype`, `research` and `brief` everywhere in commons
(one pre-existing English "brief" is allow-listed by file), and bans
`rubric` in every module added for ra. `rubric` and `report` predate ra in
the eval judge and `StageReporter`, so a blanket ban would fail on code
that was never ra's.

## Consequences

- Every new mechanism is usable by an agent with no research concept at
  all (the tests exercise them with neutral data).
- The evidence ledger treats one span of one page as one piece of
  evidence regardless of which question found it — a second sighting is
  not corroboration.
- The guard is narrower than the build brief asked (`report` is not
  banned); the reason is recorded above.

## Alternatives rejected

- **Write them inside ra.** The next agent would copy them; the repo's
  failure log lists exactly that regression.
- **Extend `llm/cascade.py` for role routing.** The cascade escalates by
  confidence; routing decides who is asked at all.
- **Reuse the Gatekeeper's cumulative ledger for budgets.** It is a
  cross-session money window, not a per-run counter.
