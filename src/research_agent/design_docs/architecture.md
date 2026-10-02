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
    [*] --> SCOPING: first message (scopable or reframe)
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
    START([start]) --> scope_gate
    scope_gate -->|Q&A| auto_freeze --> END
    scope_gate -->|Research| interview_turn
    scope_gate -->|needs reframe| await_user
    scope_gate -->|refuse| END([end])
    interview_turn --> await_user
    await_user -->|answer| interview_turn
    await_user -->|/waive| coverage_update --> interview_turn
    await_user -->|/done or yes| close_scoping
    close_scoping --> rubric_generate --> await_user
    await_user -->|/rubric …| rubric_edit --> await_user
    await_user -->|/approve| rubric_frozen --> END
    await_user -->|other command| say --> await_user
```

- **Q&A** skips the interview: `auto_freeze` builds the Brief from the
  question (and its time window), generates the `qa` rubric, freezes and
  hashes the contract, and the run graph runs it on the Q&A budget.
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
    plan_dag -->|valid| wave_start
    plan_dag -->|invalid twice| finalize
    wave_start -->|Send per ready node| research_node
    research_node --> wave_done --> wave_start
    wave_start -->|wave empty| reflect_gap_analysis
    wave_start -->|cancelled / budget| finalize
    reflect_gap_analysis -->|follow-ups| wave_start
    reflect_gap_analysis -->|stop rule met / nothing to add| synthesize
    synthesize --> independent_audit
    independent_audit -->|must-pass gaps, budget left| gap_fill
    independent_audit -->|otherwise| finalize
    gap_fill -->|targeted nodes| wave_start
    gap_fill -->|nothing to try| finalize
    finalize --> archive --> END([end])
```

- **Waves are supersteps.** `wave_start` computes the ready set from the
  commons `TaskGraph` and fans out one `Send` per node; each
  `research_node` holds a slot of the `AsyncDagScheduler`'s semaphore; a
  checkpoint lands at every wave boundary.
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
