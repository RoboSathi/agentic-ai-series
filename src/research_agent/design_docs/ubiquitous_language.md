# Ubiquitous language

These names are used as written in code, logs, stage labels and docs.

| term | meaning | in code |
|---|---|---|
| **Direct answer** | a plain question, answered in one turn by a ReAct loop over read-only tools and checked against what they returned | `graph/answerer.py` |
| **Research kind** (Archetype) | the kind of `/research` run: `decision_support`, `landscape_brief`, `fact_verification`, `explainer`, `general`; picks the checklist, rubric, stop rule and report | `domain/scope.py::Archetype`, `scoping/kind.py` (`KindClassifier` Protocol) |
| **Latest by default** | a question that names no date asks for what is current; a search pinned to a past year, or with only old results, carries a note the model reads | `research/recency.py` |
| **Time window** | "in the last 1 month" → 30 days: the Brief's `recency_days`, sent to the search providers and enforced by the worker | `scoping/window.py` |
| **Profile** | a named posture of the agent's config: live (default) or `demo` (offline Kelvari) | `config/profiles/` |
| **Search quota** | a provider's monthly request allowance, counted across processes (Tavily: 900) | `agent_platform` rate limiter |
| **Coverage checklist** | per-archetype interview items, each `open`, `covered`, `waived` or `delegated` (answered only "all" twice: the interviewer proposes values, marked `derived`, shown at rubric review) | `scoping/checklists.yaml`, `scoping/checklist.py` |
| **Known unknown** | a scope item the user waived or left open at `/done`; must be acknowledged by the rubric and the report | `domain/scope.py::KnownUnknown` |
| **Scope Brief** | the typed, frozen output of scoping: archetype, purpose/decision, criteria, constraints, source policy, recency/geography, depth, deliverable, budget, known unknowns | `domain/scope.py::ScopeBrief` |
| **Rubric** / **Rubric Item** | the acceptance criteria; each item has an id, text, a check (`deterministic`, `nli` or `judge` + name + params), a threshold, `must_pass`, an origin and what it maps to | `domain/rubric.py` |
| **Research Contract** | Brief + Rubric + Budget, frozen at `/approve` and identified by its canonical hash — the only thing the autonomous phase receives | `domain/rubric.py::Contract` |
| **Budget** | the run's hard caps: searches, fetches, tokens, seconds, rounds, DAG nodes, gap-fill rounds, concurrency | `domain/budget.py`, enforced by `agent_platform.loop_control.BudgetGuard` |
| **Plan DAG** / **sub-question** | the planner's decomposition; every node traces to at least one Rubric Item | `domain/plan.py` |
| **Round** | one plan → research → reflect cycle | `RunState.round` |
| **Batch** | one set of concurrently-runnable DAG nodes (a LangGraph superstep) | `graph/run_graph.py` |
| **Evidence Ledger** | run-scoped, append-only: claim → source URL → verbatim span → page hash → retrieved-at | `agent_platform.evidence.EvidenceLedger` |
| **Verbatim-span rule** | a span is admitted only if it is a substring of the page text this run fetched, whose hash it records | `agent_platform.evidence.VerbatimSpanValidator` |
| **Gap** | a Rubric Item the ledger (or report) does not yet satisfy | `research/reflector.py::GapAnalyzer` |
| **Stop rule** | the archetype's "done" test (every option scored; no new entities; every claim cross-checked; every concept cross-checked) | `research/stop_rules/` |
| **Independent Audit** | the independent verification; its signature accepts only `(contract, report, ledger_snapshot)` | `audit/independent.py::IndependentAuditor.audit_report` |
| **Gap-fill** | one bounded round of targeted research for must-pass evidence gaps the audit found | `graph/run_nodes.py::gap_fill` |
| **Writing fault** / **Revision** | an audit failure of how the report is written (`reflection.writing_checks`), fixed by revising the report against the audit's findings; bounded, stopped on a plateau | `graph/run_nodes.py::revise` |
| **Cross-check** | a subject supported by entailed claims from at least two independent domains, none contradicted by another | `audit/checks.py::claim_verdict` |
| **Run event** | what a finished run's critics found (failed criteria, rejection codes, limits hit), left in the dream inbox as signals scoped to the run's kind | `memory/run_events.py::RunEventWriter` |
| **Learned skill** | a finding that recurred across runs of one kind, written by the dream as a SKILL.md and active once a curator approves it in `/queue` (ra-0023) | `agent_platform.memory.dream.RecurringSignalSkills` |
| **Coverage & Limits** | the report section, built in code, that lists every unmet must-pass item with what was tried and why the run stopped | `synthesis/limits.py` |
| **COMPLETE / PARTIAL** | every must-pass item met and no budget exhausted / anything less | `graph/run_nodes.py::_final_status` |
| **Provider summary** | text a search provider writes about its results (Tavily's `answer`); untrusted, never evidence | `agent_platform.web.SearchResponse.provider_summary` |
| **Hostile page** | a page carrying instructions aimed at the reader (override, exfiltration, fake authority) | `fixture_world/world.yaml` (`hostile:`) |
