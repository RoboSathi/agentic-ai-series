# Architecture

Two LangGraph graphs, both durable through the platform's
`DurableGraphRunner` (SQLite checkpoints, `RunRegistry` rows):

- the **scoping graph** — one thread per scope; it *pauses on the user*
  every turn (`await_decision`, an `interrupt`) and `handle()` resumes it
  with the next message;
- the **run graph** — one thread per approved contract; autonomous from
  `/approve` to delivery.

`ResearchAgent.handle(query, session_id)` dispatches on the session's
phase. Every node reports through `StageReporter` (the TUI's live view)
and every transaction goes to the `AuditLedger`.

## Session phases

```mermaid
stateDiagram-v2
    [*] --> SCOPING: /research (a research kind)
    [*] --> [*]: refused
    SCOPING --> SCOPING: interview turn · /waive · challenge
    SCOPING --> RUBRIC_REVIEW: /done or confirmed proposal (rubric generated)
    RUBRIC_REVIEW --> RUBRIC_REVIEW: /rubric add | drop | set
    RUBRIC_REVIEW --> RUNNING: /approve (contract frozen + hashed)
    RUNNING --> AUDITING: report synthesized
    AUDITING --> RUNNING: must-pass gaps → one gap-fill round
    AUDITING --> DELIVERED: COMPLETE or PARTIAL
    RUNNING --> DELIVERED: PARTIAL (budget exhausted)
    RUNNING --> CANCELLED: cancel key (checkpoint kept)
    RUNNING --> FAILED: plan invalid twice
    DELIVERED --> SCOPING: a new request starts a new scope
```

With the `rubric` feature withheld there is no `RUBRIC_REVIEW` stage:
`brief_ready` asks the user to approve the Brief itself.

## Scoping graph

```mermaid
flowchart LR
    START([start]) --> research_kind
    research_kind -->|a kind| interview_turn
    research_kind -->|refuse| END([end])
    interview_turn --> await_user
    await_user -->|answer| interview_turn
    await_user -->|/waive| coverage_update --> interview_turn
    await_user -->|/done or yes| close_scoping
    close_scoping --> rubric_generate --> await_user
    await_user -->|/rubric …| rubric_edit --> await_user
    await_user -->|/approve| rubric_frozen --> END
    await_user -->|other command| say --> await_user
```

- **A plain question never reaches this graph**: it is answered directly
  (ADR ra-0020). `research_kind` decides which kind of research a
  `/research` question asks for — cue rules, the example classifier, then
  the LLM — and that kind picks the checklist the interview covers.
- **Termination is enforced in code** (`scoping/termination.py`): the
  interviewer may only *propose* done when no required item is open; the
  user confirms. `/done` closes at any time — open items become known
  unknowns. At `scoping.max_turns` the agent stops asking and offers
  `/waive <id|all>` or `/done`.
- **Anti-sycophancy guard** (`scoping_challenge_check`): at least one
  challenge or follow-up every K turns; on violation the interviewer is
  re-prompted once with an explicit challenge instruction, and an audit
  event is written.

## Run graph

```mermaid
flowchart TD
    START([start]) --> init_run --> plan_dag
    plan_dag -->|valid| batch_start
    plan_dag -->|invalid twice| finalize
    batch_start -->|Send per ready node| research_node
    research_node --> batch_done --> batch_start
    batch_start -->|batch empty| reflect_gap_analysis
    batch_start -->|cancelled / budget| finalize
    reflect_gap_analysis -->|follow-ups| batch_start
    reflect_gap_analysis -->|stop rule met / nothing to add| synthesize
    synthesize --> independent_audit
    independent_audit -->|must-pass gaps, budget left| gap_fill
    independent_audit -->|otherwise| finalize
    gap_fill -->|targeted nodes| batch_start
    gap_fill -->|nothing to try| finalize
    finalize --> archive --> END([end])
```

- **Batches are supersteps.** `batch_start` computes the ready set from the
  commons `TaskGraph` and fans out one `Send` per node; each
  `research_node` holds a slot of the `AsyncDagScheduler`'s semaphore; a
  checkpoint lands at every batch boundary.
- With `independent_audit` withheld, `synthesize` goes straight to
  `finalize`; with `reflection` withheld the plan runs once.

## One research node (the worker)

```mermaid
flowchart LR
    q[query] --> ob{outbound guard}
    ob -->|blocked| trace[(trace)]
    ob --> reserve[reserve search] --> search[WebGateway.search<br/>cache → router: DuckDuckGo / Tavily]
    search --> policy[recency window · source policy]
    policy --> fetch[reserve fetch → WebGateway.fetch]
    fetch --> screen{injection screen}
    screen -->|flagged| trace
    screen --> read[quarantined extractor]
    read --> ledger[(Evidence Ledger<br/>verbatim-span rule)]
```

Budgets are reserved **before** every await (`BudgetGuard.try_reserve` is a
synchronous check-and-decrement), so concurrent workers on one event loop
cannot overshoot a cap.

## Data flow and what each role may see

```mermaid
flowchart LR
    user((user)) -->|answers| interviewer
    interviewer --> brief[Scope Brief]
    brief --> rubric[Rubric]
    brief & rubric --> contract[[Research Contract<br/>frozen · hashed]]
    contract --> planner --> plan[Plan DAG]
    plan --> worker
    web[(web / fixtures)] -->|raw page text| extractor
    worker --> extractor --> ledger[(Evidence Ledger)]
    ledger --> reflector
    ledger --> synthesizer --> report[Report]
    contract & report & ledger --> audit{{Independent Audit<br/>isolated sub-graph}}
    audit --> final[Coverage & Limits · status]
    final --> archive[(brief · rubric · report artifacts<br/>+ run index)]
```

Only the **extractor** reads raw page text. The planner, reflector and
synthesizer see the contract, the ledger's verbatim claims and run
bookkeeping — never a page (an oracle invariant, `no_raw_page_text`). The
The independent audit sees exactly three inputs; `IsolatedGraphRunner` refuses any
other key.

## Composition root

`graph/builder.py::build_agent` is the one place features are read and
concrete classes chosen. Anything passed in is used as-is (tests pass a
replaying gateway, in-memory stores; the evals pass recording role
clients and per-case settings); anything omitted is built from config.
A withheld feature is a collaborator that is never built.
