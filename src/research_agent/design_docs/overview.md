# Research Agent — overview

An autonomous research agent that **interviews the user to bound the
scope**, derives **frozen acceptance criteria**, **researches autonomously
against that contract**, and has an **independent audit check the
result**. It is a writer/verifier split turned into a graph: the
**Elephant** is the frozen Scope Brief + Rubric that remembers what was
agreed; the **Independent Audit** is a fresh-context verifier that sees only the
contract, the report and the evidence.

It is deliberately **not an open-ended search engine**:

- no research starts without a scope the user approved (`/approve`);
- every run has hard budgets (searches, fetches, tokens, seconds, rounds,
  DAG nodes, gap-fill rounds);
- every run ends `COMPLETE`, `PARTIAL`, `CANCELLED` or `FAILED` — and a
  `PARTIAL` report says, item by item, what it could not establish and why.

## Two paths

- **A plain question** — answered directly in one turn: a ReAct loop over
  web search, page fetch, weather and the clock, checked against what the
  tools returned. No interview. A question that names no date gets the
  latest information.
- **`/research [question]`** — the full flow: a short interview, acceptance
  criteria you can edit and `/approve`, an autonomous run, an independent
  audit, a report that ends COMPLETE or PARTIAL. The research kind
  (decision, landscape, fact-check, explainer, or general) is picked for you.

Only the user chooses between them (ADR ra-0020).

## Running it

```
make agent ra                 # LIVE: real web + NVIDIA models (needs nvidia_api_key;
                            #   tavily_api_key optional — DuckDuckGo is used first)
make agent ra PROFILE=demo    # offline demo over the synthetic Kelvari world
make runs ra                # archived runs (re-verifies every artifact's hash)
make eval ra                # evals: always offline (demo profile), deterministic
make data ra                # record + validate the fixture world (offline)
```

**Live** (default): searches go through a router — DuckDuckGo first for
general questions, Tavily first for news/time-bounded ones and as the
fallback — and every role is an NVIDIA NIM model. It stays inside free
tiers: Tavily is capped at 900 calls a month (shared by every process),
searches are cached for 24h, and a research run spends ≤8 searches
(ADR ra-0018). **Demo**: the web is replayed from the recorded fixture
world and every role is the fixture world's reference model — offline and
reproducible, but it can only research the Kelvari topics the greeting lists.

## Where things are

| package | what it owns |
|---|---|
| `scoping/` | the research kind, per-archetype checklists, hybrid termination, the anti-sycophancy guard, the Brief builder |
| `rubric/` | templates per archetype, the deterministic generator, the editor, freeze + hash |
| `planning/` | the planner and the plan validation gate |
| `research/` | the worker, the three content defences, extraction, reflection, stop rules, telemetry |
| `synthesis/` | the report writer, the citation check, Coverage & Limits |
| `audit/` | the named acceptance checks and the Independent Auditor |
| `archive/` | the three-artifact archive, the run index, `make runs ra` |
| `graph/` | both LangGraph graphs, their nodes, the agent, the composition root |
| `console/` | `ResearchHost` — the agent as the shared TUI sees it |
| `fixture_world/` | the synthetic domain (tests, evals, `PROFILE=demo`), its mock web, recorder, reference model and scripted user |
| `evals/` | the eval plugin, oracle, reference judge, datasets, planted flaws, ablations |

## Further reading

[ubiquitous_language.md](ubiquitous_language.md) ·
[architecture.md](architecture.md) · [threat_model.md](threat_model.md) ·
[evals.md](evals.md) · [patterns.md](patterns.md) · [adr/](adr/)
