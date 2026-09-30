# Research Agent — overview

An autonomous research agent that **interviews the user to bound the
scope**, derives **frozen acceptance criteria**, **researches autonomously
against that contract**, and has an **independent "Goldfish" pass audit the
result**. It is Rensin's *Elephants, Goldfish…* turned into a graph: the
**Elephant** is the frozen Scope Brief + Rubric that remembers what was
agreed; the **Goldfish** is a fresh-context verifier that sees only the
contract, the report and the evidence.

It is deliberately **not an open-ended search engine**:

- no research starts without a scope the user approved (`/approve`);
- every run has hard budgets (searches, fetches, tokens, seconds, rounds,
  DAG nodes, gap-fill rounds);
- every run ends `COMPLETE`, `PARTIAL`, `CANCELLED` or `FAILED` — and a
  `PARTIAL` report says, item by item, what it could not establish and why.

## Running it

```
make data ra        # record + validate the fixture world (idempotent)
make tui ra         # interview → rubric → autonomous run → audited report
make runs ra        # archived runs (re-verifies every artifact's hash)
make eval ra        # generate → oracle → security → judge → deepeval
```

By default the web is **replayed** from the recorded fixture world and
every model role is the fixture world's **reference model**
(`research_agent.model_backend: fixture`), so the whole flow runs offline
and reproducibly. `model_backend: routed` sends each role to the model in
`llm.roles`; `web.mode: live` (with `tavily_api_key`) uses the real web.

## Where things are

| package | what it owns |
|---|---|
| `scoping/` | the scopability gate, per-archetype checklists, hybrid termination, the anti-sycophancy guard, the Brief builder |
| `rubric/` | templates per archetype, the deterministic generator, the editor, freeze + hash |
| `planning/` | the planner and the plan validation gate |
| `research/` | the worker, the three content defences, extraction, reflection, stop rules, telemetry |
| `synthesis/` | the report writer, the citation check, Coverage & Limits |
| `audit/` | the named acceptance checks and the Goldfish Auditor |
| `archive/` | the three-artifact archive, the run index, `make runs ra` |
| `graph/` | both LangGraph graphs, their nodes, the agent, the composition root |
| `console/` | `ResearchHost` — the agent as the shared TUI sees it |
| `fixture_world/` | the synthetic domain, its mock web, recorder, reference model and scripted user |
| `evals/` | the eval plugin, oracle, reference judge, datasets, planted flaws, ablations |

## Further reading

[ubiquitous_language.md](ubiquitous_language.md) ·
[architecture.md](architecture.md) · [threat_model.md](threat_model.md) ·
[evals.md](evals.md) · [patterns.yaml](patterns.yaml) · [adr/](adr/) ·
[BUILD_PLAN.md](BUILD_PLAN.md) (reuse map, deviations)
