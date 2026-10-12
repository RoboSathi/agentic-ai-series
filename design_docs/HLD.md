# HLD — Agentic Harness Platform

The high-level design: what the system is made of, how an agent is composed from the
platform, and how a request flows through it. Component internals are in [LLD.md](LLD.md);
the decisions behind them in [ADR.md](ADR.md); terms in
[ubiquitous_language.md](ubiquitous_language.md).

## 1. Architectural style

A **monorepo, modular-monolith runtime**. One Python package, `agent_platform`, is a shared
library of narrow, independently testable subsystems. Three agents compose it:

| Agent | Package | What it does |
|---|---|---|
| Customer Support (`csa`) | `src/customer_support_agent/` | Answers order and refund questions, issues refunds within its authority, queues larger ones for a person, refuses what policy forbids |
| Research (`ra`) | `src/research_agent/` | Answers a plain question directly with web tools; `/research` runs a scoped, contract-bound, independently audited research run |
| Ops Analyst (`ops`) | `src/ops_agent/` | Answers questions about how the agents are running, using only the platform's analytics MCP server |

Each agent runs in its own process inside the platform's Textual console (`make agent <agent>`).
Orchestration inside an agent is **LangGraph**; the platform does not re-implement state
management LangGraph provides. Everything downstream of the model is the **harness**: the
controls you can remove and still have an agent that runs (the declared features), plus the
plumbing that is always on.

## 2. System context

```mermaid
flowchart TB
    subgraph Operator
        TUI["Textual console<br/>(make agent &lt;agent&gt;)"]
        MAKE["make targets<br/>(audit, review, dream, eval, …)"]
        RILL["Rill dashboards<br/>(make dashboard)"]
        EXT["External MCP client<br/>(Claude Code, Inspector)"]
    end
    subgraph Runtime["agentic-harness-core (one process per agent)"]
        AGENT["Agent graph<br/>(LangGraph)"]
        AP["agent_platform"]
        AGENT -->|composes| AP
    end
    subgraph Hosted["Hosted services"]
        NIM["NVIDIA NIM<br/>chat · embeddings · guards"]
        WEB["Web search<br/>DuckDuckGo · Tavily"]
        LS["LangSmith<br/>(optional)"]
    end
    subgraph Local["Local machine"]
        LM["local_models/<br/>NLI · reranker · intent · summarizer · spaCy · embedder"]
        DB[("db/&lt;package&gt;/<br/>SQLite · DuckDB")]
        DATA[("data/&lt;package&gt;/<br/>memory · runs · Parquet")]
        LOGS[("logs/&lt;agent&gt;/")]
    end
    TUI <--> Runtime
    MAKE --> AP
    AP --> NIM
    AP --> WEB
    AP --> LS
    AP --> LM
    AP --> DB
    AP --> DATA
    AP --> LOGS
    RILL --> DATA
    EXT -->|stdio| MCP["analytics MCP server"]
    MCP --> DATA
```

## 3. Platform component map

`agent_platform` keeps its well-known subsystems at the top level and groups small leaves into
four packages. The order below is the order of importance for reading the code.

```mermaid
flowchart TB
    subgraph Interface["interface/ — operator surface"]
        tui & commands & presentation & host
    end
    subgraph Controls["Harness controls (declared features)"]
        security & deterministic_checks & intent & gatekeeper
        hallucination_check & guardrails & memory & context
    end
    subgraph Loop["Running the agent"]
        loop_control & tools
        subgraph Runtime["runtime/"]
            executor & scheduling & verification
        end
        durability
    end
    subgraph Models["Model access"]
        llm & judge & scoring
    end
    subgraph Knowledge["knowledge/"]
        web & mcp & artifacts
    end
    subgraph Plumbing["Always on"]
        resilience & observability & audit & review & logging
        subgraph Core["core/"]
            persistence & primitives
        end
        config
    end
    subgraph Offline["Offline"]
        evals & analytics
    end
    Interface --> Controls --> Loop --> Models
    Loop --> Knowledge
    Controls --> Plumbing
    Loop --> Plumbing
    Models --> Plumbing
    Offline --> Plumbing
```

| Group | Package | Responsibility |
|---|---|---|
| — | `config` | Layered YAML → typed settings; the `FeatureSet`; `REPO_ROOT`-relative paths |
| — | `llm` | The only place a provider is called: adapters, `ModelChainBuilder`, rate-limited HTTP, local encoders, metering, PII boundary hook |
| — | `gatekeeper` | Deterministic authority decisions with standings; mandates; RBAC; authority rank |
| — | `tools` | Typed tools, namespaced registry, retrieval-based selection, the single tool path |
| — | `loop_control` | `LoopGovernor`: step cap, budgets, deadline, stagnation, convergence; one `LoopVerdict` |
| — | `hallucination_check` | Claim-level grounding in levels; fact ledger; state model; evidence ledger |
| — | `guardrails` | Structured-output validation and repair; the response guardrail |
| — | `security` | RBAC policy, PII (Presidio + vault), prompt injection, content moderation, guard chains |
| — | `memory` | The four CoALA stores as files; governed writes; dreams; skills; RAG |
| — | `context` | Five-stage context assembly, strategies, failure detection |
| — | `intent` | Intent classifier contracts; Laya and BERT providers |
| — | `deterministic_checks` | The pre-LLM check contract (the business-logic seam) |
| — | `durability` | Checkpointed runs, resume, HITL interrupts, isolated sub-graphs, run controls |
| — | `resilience` | Breaker, retry, timeout, bulkhead, idempotency, saga, hedge, admission, pub/sub, DLQ, rate limiter |
| — | `observability` | Status feed, cards, facts, task/turn bracket, SLOs, cost, LangSmith export |
| — | `audit` | Append-only decision trail; the audit view |
| — | `review` | One human-review queue for every agent |
| — | `logging` | Per-subsystem file logs; `@instrument`; trace id |
| — | `judge` | The shared LLM judge (Template Method) and calibration |
| — | `scoring` | The NOLLM scoring cascade and entailment models |
| — | `evals` | Oracle, DeepEval and Promptfoo graders; intent bake-off; eval store |
| — | `analytics` | Semantic layer, Parquet export, metric catalogue, dummy world, generated Rill dashboards |
| `core/` | `persistence`, `primitives` | SQLite pool and platform tables; money, hashing, keys, PII tokens, clock |
| `interface/` | `tui`, `commands`, `presentation`, `host` | The console, slash commands, plain-text renderers, the `AgentHost` contract |
| `runtime/` | `executor`, `scheduling`, `verification` | Running approved calls; DAG scheduling; effect read-back and acceptance checks |
| `knowledge/` | `web`, `mcp`, `artifacts` | Web gateway (live/record/replay), MCP client and servers, frozen hashed artifacts |

## 4. How an agent is composed

Every agent follows the same composition pattern: a config overlay declares what it is, one
composition root (`graph/builder.py::build_agent`) reads the feature tree once and builds only
what is declared, and an `AgentHost` hands the result to the console.

```mermaid
flowchart LR
    subgraph Config["Configuration (read once)"]
        P["agent_platform/config/config.yaml<br/>(platform defaults)"]
        O["src/&lt;agent&gt;/config/config.yaml<br/>(overlay: identity, features, models, paths)"]
        PR["profiles/&lt;name&gt;.yaml<br/>(PROFILE=…)"]
        L["config/local.yaml · .env · SECTION__FIELD"]
        P --> M{{"deep merge"}}
        O --> M
        PR --> M
        L --> M
        M --> S["AppConfig + FeatureSet"]
    end
    subgraph Root["Composition root — build_agent()"]
        S --> B["read features once"]
        B --> C1["build declared controls<br/>(screens, checks, intent, gatekeeper,<br/>grounding, memory, context)"]
        B --> C2["always-on plumbing<br/>(model chain, tool path, executor,<br/>loop governor, audit, facts, durability)"]
        C1 --> G["LangGraph StateGraph<br/>(withheld stage = no node)"]
        C2 --> G
    end
    subgraph Shell["Console"]
        H["AgentHost<br/>(agent.host in config)"] --> T["AgenticTuiApp"]
        G --> H
        T --> CMD["platform commands + agent commands"]
    end
```

- **Overlays** are named from outside: the Makefile exports `AGENTIC_OVERLAYS` for the agent
  named on the command line. The platform's own `overlays:` is empty.
- **Injection over construction.** Every collaborator of `build_agent` is a keyword argument
  (`llm=`, `pool=`, `gatekeeper=`, `audit=`, `reporter=`, `checkpointer=`, …); anything passed is
  used as is, anything omitted is built from config. Tests and the eval harness substitute what
  they need.
- **A withheld feature is a stage that is never built.** The graph's edges are wired to the next
  stage that exists, so the trace has a gap where the control used to be.
- **`/harness` hot-reloads** the feature tree and rebuilds the agent in the running console;
  nothing is written to disk.

## 5. The three agents

### 5.1 Customer Support

```mermaid
flowchart LR
    START([turn]) --> authenticate
    authenticate -->|unverified| END1([reply])
    authenticate --> choose_order
    choose_order -->|menu choice answered| END1
    choose_order --> pii_filtering
    pii_filtering --> prompt_injection --> content_moderation --> deterministic_checks
    deterministic_checks --> classify_intent
    classify_intent -->|deterministic read| read_intent --> END1
    classify_intent --> assemble_context --> prefetch --> propose
    propose -->|tool calls| execute
    execute -->|continue| propose
    propose -->|no tool call / cap| finalize
    execute -->|refusal final / escalated| finalize
    finalize -->|queued for a person| await_approval --> END2([paused, checkpointed])
    finalize --> END1
```

- Every stage between `choose_order` and `prefetch` exists only when its feature is declared
  (`pii`, `prompt_injection`, `content_moderation`, `deterministic_checks`,
  `intent_classification`, `context`); any of them can halt the turn.
- `propose ⇄ execute` is the **tool-use loop**: the model proposes typed calls; the tool path
  (registry → loop governor → Gatekeeper → Executor) decides whether they happen. It is
  deliberately not ReAct.
- `finalize` runs the response guardrail (schema, then the hallucination check). A reply the
  check cuts, and any refund the Gatekeeper escalates, become review items.
- Every turn is a durable run; `await_approval` is a LangGraph `interrupt` resumed by the
  reviewer's decision.

### 5.2 Research

Two paths, chosen by the user (ADR R-01): a plain question goes to the **answer graph**;
`/research` goes to the **scoping graph** and then the **run graph**.

```mermaid
stateDiagram-v2
    [*] --> ANSWER: plain question
    ANSWER --> [*]: grounded answer
    [*] --> SCOPING: /research
    SCOPING --> SCOPING: interview turn · /waive · challenge
    SCOPING --> RUBRIC_REVIEW: /done or confirmed proposal
    RUBRIC_REVIEW --> RUBRIC_REVIEW: /rubric add | drop | set
    RUBRIC_REVIEW --> RUNNING: /approve (contract frozen + hashed)
    RUNNING --> AUDITING: report synthesized
    AUDITING --> RUNNING: evidence gap → gap-fill
    AUDITING --> AUDITING: writing fault → revise
    AUDITING --> DELIVERED: COMPLETE or PARTIAL
    RUNNING --> DELIVERED: PARTIAL (budget exhausted)
    RUNNING --> STOPPED: cancel (checkpoint kept)
    STOPPED --> RUNNING: /resume
    STOPPED --> [*]: /abort
```

```mermaid
flowchart TD
    subgraph Answer["Answer graph"]
        a_reason[reason] -->|tool calls| a_act[act] --> a_reason
        a_reason --> a_ground[ground] --> a_fin[finalize]
        a_reason --> a_fin
    end
    subgraph Run["Run graph"]
        init_run --> plan_dag
        plan_dag -->|valid| batch_start
        plan_dag -->|invalid twice| finalize
        batch_start -->|Send per ready node| research_node --> batch_done --> batch_start
        batch_start -->|batch empty| reflect_gap_analysis
        reflect_gap_analysis -->|follow-ups| batch_start
        reflect_gap_analysis --> synthesize --> independent_audit
        independent_audit -->|writing fault| revise --> independent_audit
        independent_audit -->|evidence gap| gap_fill -->|targeted nodes| batch_start
        independent_audit --> finalize --> archive
    end
```

- Scoping pauses on the user every turn (`await_user`, an `interrupt`); termination is enforced
  in code (R-03). The rubric is generated deterministically (R-04).
- Batches are supersteps: one `Send` per ready DAG node, each under the scheduler's semaphore;
  budgets reserved before every await (R-06).
- Only the extractor reads raw page text; the planner, reflector and synthesizer see the
  contract and the ledger's verbatim claims (R-07, R-08).
- The independent audit runs in an isolated sub-graph that accepts exactly
  `(contract, report, ledger_snapshot)` and a judge on a different model (R-11).

### 5.3 Ops Analyst

```mermaid
flowchart LR
    START([question]) --> screen{prompt injection<br/>screen}
    screen -->|blocked| END1([refusal])
    screen --> propose
    propose -->|MCP tool calls| execute --> propose
    propose --> ground["ground<br/>(logprobs → figure check)"] --> END2([answer + SQL + rows])
```

Its only tools are the analytics MCP server's (`list_metrics`, `compute_metric`, `list_views`,
`describe_view`, `run_query`), arriving through the governed MCP client and the same tool path
as any local tool.

## 6. Request lifecycle

### 6.1 A refund turn (Customer Support)

```mermaid
sequenceDiagram
    autonumber
    actor C as Customer
    participant TUI as Console
    participant G as CSA graph
    participant SEC as security (PII · injection · moderation)
    participant DC as OrderChecks
    participant IR as intent router
    participant CTX as context + skills
    participant LLM as model chain
    participant TP as tool path
    participant GK as Gatekeeper
    participant EX as Executor + RefundSaga
    participant VER as verification
    participant HC as hallucination check
    participant RQ as review queue
    participant OBS as audit · facts · logs

    C->>TUI: "refund order 10693"
    TUI->>G: handle(query, session) — durable thread
    G->>G: authenticate (verified session) · choose_order
    G->>SEC: redact PII · screen injection · moderate
    SEC-->>G: pass (each a check fact)
    G->>DC: order exists, is the customer's, refundable?
    DC-->>G: findings (no model call)
    G->>IR: phrase rules → Laya classifier
    IR-->>G: REQUEST_REFUND (write)
    G->>CTX: assemble context · select skill refund-workflow
    G->>TP: prefetch get_order_status, get_refund_status
    TP->>GK: authorize reads → VALID
    TP->>EX: run reads
    G->>LLM: propose (offered: process_refund)
    LLM-->>G: tool call process_refund(order_id)
    G->>TP: proposal
    TP->>GK: offered? mandate? RBAC? within limit?
    alt within the refund limit
        GK-->>TP: allow · VOIDABLE
        TP->>EX: execute (idempotency key, deadline)
        EX-->>TP: refund issued
        TP->>VER: read the refund back
        VER-->>TP: verified
    else above the limit
        GK-->>TP: escalate · RATIFIABLE
        TP->>RQ: queue refund for a person
    end
    TP->>OBS: audit entry · fact_tool_call
    G->>LLM: write the reply
    LLM-->>G: draft
    G->>HC: level 1 logprobs → level 2 grounding
    HC-->>G: verdicts (cut / correct / keep)
    G-->>TUI: reply (+ paused for review if escalated)
    G->>OBS: turn fact · LangSmith trace · dream event
```

### 6.2 A research run

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant RA as Research agent
    participant SC as scoping graph
    participant RG as run graph
    participant W as research workers
    participant WEB as web gateway
    participant L as evidence ledger
    participant AU as independent audit
    participant AR as archive

    U->>RA: /research <question>
    RA->>SC: research kind (cues → example classifier → LLM)
    loop interview turns
        SC-->>U: question (checklist item)
        U->>SC: answer / /waive / /done
    end
    SC->>SC: generate rubric (deterministic)
    U->>SC: /rubric edits, then /approve
    SC->>RG: frozen, hashed contract
    RG->>RG: plan DAG (validated)
    loop batches
        RG->>W: Send per ready node
        W->>WEB: search (cache → router) · fetch (budget reserved first)
        WEB-->>W: pages (outbound guard, source policy, injection screen)
        W->>L: verbatim spans only
        RG->>RG: reflect: gaps → follow-ups or stop rule
    end
    RG->>RG: synthesize report (citation guard)
    RG->>AU: (contract, report, ledger_snapshot)
    AU-->>RG: verdicts: deterministic → NLI → judge
    alt writing fault
        RG->>RG: revise (bounded)
    else evidence gap
        RG->>W: one gap-fill round
    end
    RG->>AR: brief · rubric · report (hashed) · run dream event
    RG-->>U: COMPLETE or PARTIAL + Coverage & Limits
```

### 6.3 An ops question over MCP

```mermaid
sequenceDiagram
    autonumber
    actor O as Operator
    participant OPS as Ops agent
    participant LLM as model chain
    participant TP as tool path + Gatekeeper
    participant MC as governed MCP client
    participant MS as analytics MCP server
    participant PQ as Parquet (analytics source)

    O->>OPS: "p95 model latency by role this week?"
    OPS->>OPS: screen for injection · skill analytics-investigation
    OPS->>MC: prefetch list_metrics, list_views
    loop until answered (max_steps)
        OPS->>LLM: propose
        LLM-->>OPS: compute_metric / run_query
        OPS->>TP: validate · authorize (ops_agent.policy) · execute
        TP->>MC: call tool
        MC->>MS: stdio
        MS->>MS: QueryGuard (one SELECT over curated views)
        MS->>PQ: in-memory DuckDB over Parquet
        PQ-->>MS: rows
        MS-->>MC: result (or a refusal that says what to change)
    end
    OPS->>OPS: logprobs check → figure check (every number in a result)
    OPS-->>O: answer + the SQL + the rows
```

## 7. Cross-cutting concerns

- **The platform names no agent.** It reaches agents through `AgentHost`, `EvalPlugin` and
  config; guard tests enforce it (ADR P-01).
- **Authority is decided, not inferred.** The Gatekeeper reads trusted policy and tool metadata,
  never source plausibility; a record fact belongs to the tool, an authority fact to the
  Gatekeeper (P-09).
- **Exposure is not authorization.** Which tools a turn offers (skills, intent) and whether a
  proposed call may run (Gatekeeper) are separate decisions.
- **A side effect is not believed until it is read back** (P-18).
- **No action is un-audited.** The audit entry is written on every branch, including denials and
  UNGOVERNED calls (P-10).
- **No control state is decorative.** Nothing is logged or shown that no code path produces or
  that cannot change a verdict.
- **The console is a consumer.** File logging (`logs/<agent>/<subsystem>.log`, full payloads,
  never truncated by default) is independent of what is on screen.
- **Everything is configuration.** No value is hard-coded; every path resolves against the repo
  root.

## 8. Deployment view

Single-process, local deployment per agent; several agents (and the MCP server, evals, Rill) may
run at once and share the cross-process rate-limit store.

```mermaid
flowchart LR
    subgraph P1["make agent csa"]
        T1["Textual app"] <-. one asyncio loop .-> G1["CSA graph"]
    end
    subgraph P2["make agent ops"]
        T2["Textual app"] <-.-> G2["Ops graph"]
        G2 -->|stdio| MS["analytics MCP server<br/>(child process)"]
    end
    subgraph P3["make eval csa"]
        EV["eval pipeline"]
    end
    RL[("db/agent_platform/rate_limits.sqlite<br/>shared request budget")]
    OBS[("db/agent_platform/observability.duckdb<br/>one writer")]
    PQ[("data/agent_platform/analytics/*.parquet")]
    P1 --> RL
    P2 --> RL
    P3 --> RL
    P1 -->|facts, audit| OBS
    OBS -->|export every N s| PQ
    MS --> PQ
    RILL["Rill (make dashboard)"] --> PQ
    P1 -->|HTTPS| NIM["NVIDIA NIM"]
```

- DuckDB admits one writer: the running console exports Parquet on an interval, and every reader
  (MCP server, Rill) reads the Parquet. Evals use their own store (`evals.duckdb`).
- There is no container isolation; tool execution is in-process (the Docker sandbox is parked —
  [PRD §6](PRD.md#6-parked-and-deferred)).

## 9. Technology stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph (`StateGraph`, `Send`, `interrupt`, SQLite checkpointer, AES-encrypted) |
| Hosted models | NVIDIA NIM (OpenAI-compatible): chat, embeddings, jailbreak and content-safety guards |
| Local models | Hugging Face Transformers + PyTorch (NLI, reranker, BART summarizer, BERT intent, e5 embedder), Laya decision model, spaCy `en_core_web_lg` for Presidio |
| Web | DuckDuckGo (`ddgs`), Tavily; HTTP fetcher with a reader fallback |
| Persistence | SQLite (agent databases, checkpoints, memory index, rate limits, vector search via sqlite-vec), DuckDB (observability, evals), Parquet (analytics export) |
| Protocols | MCP (official SDK) |
| Console | Textual + Rich, one app |
| Config | pydantic-settings, layered YAML, `SECTION__FIELD` env overrides |
| Tracing | LangSmith (optional), plus the repo's own logs, facts and audit trail |
| Dashboards | Rill (repo-local binary) |
| Evals | DeepEval (optional extra), Promptfoo (pinned, via `npx`) |
| Quality | ruff, mypy `--strict`, pytest with an 80% coverage gate |

## 10. Dependency direction

Dependencies point inward; a lower layer receives a higher capability by injection, never import.

```
core.primitives, config, logging                      layer 0
   └─ core.persistence, llm, resilience                 layer 1
        └─ security, memory, tools, audit, review       layer 2
             └─ context, scoring, guardrails,
                hallucination_check, intent             layer 3
                  └─ gatekeeper, runtime, loop_control,
                     durability, knowledge              layer 4
                       └─ observability, analytics      cross-cutting (read by any layer)
                            └─ evals, judge             offline
                                 └─ interface           consumers only
                                      └─ agents (customer_support_agent, research_agent, ops_agent)
```

`hallucination_check → context → tools → guardrails → hallucination_check` is a known import
cycle broken at call sites; `scoring` is kept a leaf so it stays out of it.
