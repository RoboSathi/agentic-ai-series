# Program Design — engineering standards, layout and process

How this repository is organised and how work in it is done. Requirements are in
[PRD.md](PRD.md), architecture in [HLD.md](HLD.md), decisions in [ADR.md](ADR.md), terms in
[ubiquitous_language.md](ubiquitous_language.md).

## 1. Repository layout

```
.
├── Makefile                      every operator and developer command (make help)
├── pyproject.toml · uv.lock      the package, its extras (dev, evals) and tool config
├── .env.example                  provider keys and secrets (copy to .env)
├── README.md                     setup, usage, architecture tour
├── design_docs/                  every design document (this folder)
│   └── images/                   screenshots referenced by README.md
├── src/
│   ├── agent_platform/           the shared platform (names no agent)
│   │   ├── config/ logging/ llm/ judge/ scoring/
│   │   ├── memory/ context/ tools/ loop_control/ durability/
│   │   ├── security/ intent/ deterministic_checks/ gatekeeper/
│   │   ├── hallucination_check/      (+ evidence/: tool observations, evidence ledger)
│   │   ├── guardrails/ resilience/ observability/ audit/ review/
│   │   ├── evals/
│   │   ├── analytics/                (+ dashboards/: Rill project, install/serve scripts)
│   │   ├── core/        primitives/ persistence/
│   │   ├── interface/   tui/ commands/ presentation/ host/
│   │   ├── runtime/     executor/ scheduling/ verification/
│   │   └── knowledge/   web/ mcp/
│   ├── customer_support_agent/   agent: config, graph, domain, tools, gatekeeper rules, evals, …
│   ├── research_agent/           agent: config, graph, scoping, rubric, planning, research, audit, …
│   └── ops_agent/                agent: config, graph, guardrails (figure check), prompts
├── tests/
│   ├── conftest.py               offline fixtures (scripted LLM, in-memory DB, no provider keys)
│   ├── agent_platform/           platform unit tests and guard tests
│   ├── customer_support_agent/   ┐
│   ├── research_agent/           ├ agent unit and flow tests
│   ├── ops_agent/                ┘
│   └── integration/              cross-subsystem invariants, call flows, attacks, eval pipeline
├── local_models/                 downloaded models (gitignored; make models)
├── db/<package>/                 every database (gitignored)
├── data/<package>/               memory, web cache, analytics Parquet (gitignored)
├── reports/<package>/evals/      eval reports (gitignored; folders kept)
└── logs/<agent>/                 per-subsystem log files (gitignored)
```

Rules (enforced by guard tests where noted):

- No module sits loose at a package root; only `__init__.py` and `__main__.py`
  (`test_package_layout.py`).
- `agent_platform` imports and names no agent, in code, config or documents
  (`test_platform_boundary.py`, `test_roles_and_boundary.py`).
- An agent's `domain/` never imports its `persistence/`; no import cycles between an agent's
  subpackages (`test_agent_layering.py`).
- Runtime state never lives under `src/`; every path to it is a config value resolved against the
  repo root (`config/paths.py`).
- Tests mirror the package they test: `tests/<package>/test_<subject>.py`.
- Design knowledge lives in `design_docs/`; a package's `README.md` is a short pointer (what it
  is, its entry points, the design section to read).

### Anatomy of an agent package

| Subpackage | Role |
|---|---|
| `config/` | The agent's overlay (`config.yaml`), profiles, and its own settings section |
| `graph/` | The LangGraph graphs, nodes, wiring, and `builder.py::build_agent` (the composition root) |
| `console/` | Its `AgentHost` and agent-owned slash commands |
| `domain/` | Its vocabulary and business rules; `ports.py` for persistence it needs |
| `prompts/` | `AGENT.md` (profile), `skills/*/SKILL.md`, role or reply prompts |
| `analytics/` | Its declared outcomes, Tier-2 facts, semantic layer, dummy-world profile |
| others | Its side of a platform seam, named after the platform package it plugs into (`gatekeeper/`, `memory/`, `review/`, `deterministic_checks/`, `verification/`, …) |

### Adding an agent

1. `src/<agent>/` with `config/config.yaml` (identity, `agent.host`, `features:`, paths under
   `db/<agent>/` and `data/<agent>/`), `graph/builder.py::build_agent`, `console/` (an `AgentHost`).
2. Three lines in the Makefile registry (`AGENTS`, `<name>_PKG`, `<name>_OVERLAY`, and the
   optional `_DATA`, `_RUNS`, `_EVALS`).
3. Optionally: an `EvalPlugin` (`evals.plugin`), declared outcomes and a semantic layer, a
   `MemoryProfile`, skills.

Nothing in `agent_platform` changes.

## 2. Design process — follow this order

1. **Ubiquitous language first.** Before designing a module, establish its vocabulary in
   [ubiquitous_language.md](ubiquitous_language.md). Ask rather than assume terms.
2. **Deep-module design.** Fewer, well-hidden abstractions behind small interfaces. Before adding
   a public method, ask whether it leaks a detail the caller should not need.
3. **TDD as the coding discipline**, with test-design decisions made explicitly before code:
   - **Unit granularity** — one test class per public behaviour group, one focus per test.
   - **What gets mocked** — only true I/O boundaries (model HTTP, databases, filesystem, external
     APIs). Never mock a sibling platform module in a test that exists to prove their
     integration; that belongs in `tests/integration/`.
   - **What is under test** — the interface's contract, not the implementation.
4. **Interview-style planning.** For a new module or feature, resolve one design decision at a
   time, branch by branch, before presenting a finished plan. Ask before building anything
   non-trivial.
5. **Record the decision** in [ADR.md](ADR.md) and update the affected design sections.

## 3. Coding standards checklist

- [ ] Methods roughly 5–20 lines; extract beyond that.
- [ ] One concern per module; dependencies point inward ([HLD §10](HLD.md#10-dependency-direction)).
- [ ] Every swappable component (provider, store, database) behind a Protocol or ABC, received by
      injection; no caller instantiates a concrete class it could be handed.
- [ ] No hard-coded value — no strings, numbers, model or vendor names in code; everything from
      configuration. Card headings never name a model.
- [ ] No CWD-relative paths; resolve against `REPO_ROOT`.
- [ ] Features read once, in the composition root; a withheld feature is never built.
- [ ] No orphaned modules; every module is imported from somewhere real.
- [ ] Every Protocol has a concrete implementation, verified by `mypy --strict`.
- [ ] No config field without the code that reads it, and no code path that logs or shows a
      control state that cannot change a verdict.
- [ ] Full payloads in the log files; the console is a summary, never the record.
- [ ] Human-readable CLI and console output as box-drawn tables or trees, rendered from plain
      lines shared by `make` and the console.
- [ ] Runtime artifacts never versioned.
- [ ] LLD patterns (strategy, adapter, factory, builder, observer, facade, decorator, template
      method) only where they genuinely fit.

## 4. Known failure patterns — do not repeat

- Agent code duplicating the platform (its own model client, retries around a model call, auth
  checks, token counting, print-based logging) instead of composing it.
- Orphaned or dead files left behind after a refactor.
- A Protocol declared but not implemented by its concrete classes.
- CWD-relative paths that break outside the repo root.
- Config fields with no adapter or reader behind them.
- Runtime artifacts (`.coverage`, caches, databases, logs, reports) committed.
- A copy of a business rule in the prompt standing in for the rule in code (the model obeys the
  copy, so withholding the control changes nothing).
- Calling work done from reading code — the bar is ruff, `mypy --strict` and pytest with coverage,
  run for real, plus a smoke run of every agent's golden path after a platform change.
- Tests that silently call a live provider (slow, flaky, and they spend the free tier).

## 5. Verification and CI

`make verify` is the definition of "verified":

1. `ruff check` and `ruff format --check` over `src/` and `tests/`
2. `mypy --strict src/`
3. `pytest tests/ --cov=src` with `fail_under = 80` (pyproject), offline
4. Each agent's composition root imported in its own process

There is no hosted CI workflow yet; when one is added it runs exactly `make verify` (after
`make install` without the model download), with no provider keys.

## 6. Testing strategy

See [evals.md §10](evals.md#10-tests-are-not-evals) for the levels. In short: unit tests per
module with scripted models and in-memory databases; integration tests for cross-subsystem
invariants and full call flows; guard tests for the structural rules; evals for behaviour against
real models.

## 7. Build state

| Area | State |
|---|---|
| Platform | Built: every package in §1. Parked: sandbox, multi-agent, prompt compilation ([PRD §6](PRD.md#6-parked-and-deferred)) |
| Customer Support | Built, production grade; full eval suite |
| Research | Built: direct answer and `/research`; no eval suite |
| Ops | Built: MCP demonstration agent; no eval suite |
| Episodes | Demonstrated by configuration: an agent run under its features and profiles (ADR P-06) |
