# Architecture Decision Record

One record of every decision that shapes the code as it stands today. It replaces the
separate platform ADRs (`0001`–`0013`) and research-agent ADRs (`ra-0001`–`ra-0024`): each
entry below names the original records it consolidates under **Supersedes**, so a reader of
old commits, logs or episode notes can still find the decision.

- Each entry: **Context** (the force), **Decision** (what holds now), **Consequences** (what
  follows, including what is owed), and where useful **Alternatives rejected**.
- Only the *current* decision is written. Earlier versions are listed once in
  [Appendix A — superseded and retired decisions](#appendix-a--superseded-and-retired-decisions).
- Numbering: `P-` platform, `R-` research agent, `C-` customer-support agent, `O-` ops agent.
- A change to the design adds or amends an entry here; nothing else is the decision of record.

## Index

| Group | Entries |
|---|---|
| [Platform boundary and layout](#1-platform-boundary-and-layout) | P-01 – P-06 |
| [The harness and its features](#2-the-harness-and-its-features) | P-07 – P-08 |
| [Authority: Gatekeeper, Executor, standings](#3-authority-gatekeeper-executor-standings) | P-09 – P-11 |
| [Memory, skills and human review](#4-memory-skills-and-human-review) | P-12 – P-15 |
| [Grounding and hallucination](#5-grounding-and-hallucination) | P-16 – P-18 |
| [Models, resilience and metering](#6-models-resilience-and-metering) | P-19 – P-23 |
| [Analytics, MCP and dashboards](#7-analytics-mcp-and-dashboards) | P-24 – P-27 |
| [Evals and tests](#8-evals-and-tests) | P-28 – P-29 |
| [Research agent](#9-research-agent) | R-01 – R-18 |
| [Customer-support agent](#10-customer-support-agent) | C-01 – C-07 |
| [Ops agent](#11-ops-agent) | O-01 |

---

## 1. Platform boundary and layout

### P-01 — The platform names no agent

**Context.** The platform once could not run a second agent without being edited: the TUI
built the support agent directly, and half the console commands imported its repositories.

**Decision.** Dependencies point agent → platform, never back.
- An agent reaches the console through `AgentHost` (`interface/host/`), resolved from
  `agent.host` in the agent's config — never imported.
- The agent's config layer is named from outside (`AGENTIC_OVERLAYS`, exported by the
  Makefile); the platform's own `overlays:` is empty.
- An agent joins evals through one seam (`EvalPlugin`, named in `evals.plugin`).
- Anything that needs to say "order", "customer" or "refund" is in the wrong package.

**Consequences.** `tests/agent_platform/test_platform_boundary.py` greps every module, config
file and document under `agent_platform` for an agent name; `test_roles_and_boundary.py` bans
research vocabulary (`archetype`, `research`, `brief`, and `rubric` in the modules added for
it). Adding an agent is three Makefile registry lines and nothing in the platform.

**Supersedes.** The 2026-09-24 decoupling refactor; ADR 0003 (vocabulary guard).

### P-02 — Composition is agent-owned; the platform holds mechanisms

**Context.** The research agent composes a planner, a ReAct worker, reflection and an
independent audit; the support agent composes a tool-use loop, a router and a quality gate.
The compositions look alike on a diagram and share almost no code.

**Decision.** Pattern **composition** lives in the agent's graph. Only domain-free
**mechanisms** go into `agent_platform`: a DAG scheduler, a budget guard, an isolated
sub-graph runner, a role router, an evidence ledger, a web gateway, an acceptance-check
registry. A mechanism is promoted only when a second agent needs the identical thing, and
then behind a Protocol.

**Consequences.** A new agent gets mechanisms, not a framework. Mechanisms added for the
research agent (each unit-tested with neutral data): `runtime/scheduling` (`TaskGraph`,
`AsyncDagScheduler`), `loop_control/budget.py` (`BudgetGuard`), `knowledge/artifacts`,
`hallucination_check/evidence` (`EvidenceLedger`, `VerbatimSpanValidator`), `knowledge/web`,
the monthly quota in `resilience/rate_limiter`, `llm/roles.py`, `durability/isolated.py`,
`durability/hitl.py`, `runtime/verification/acceptance.py`, `core/primitives/hashing.py`.

**Alternatives rejected.** Promote each pattern to the platform as a reusable node (every node
would carry the first agent's assumptions); write mechanisms inside the agent (the next agent
copies them — a regression this repo has had).

**Supersedes.** ADR 0001, ADR 0003.

### P-03 — Package layout: grouped subsystems, runtime state outside `src/`

**Context.** `agent_platform` had grown to forty top-level packages, many of them small
leaves, and databases, memory files and reports lived inside the source tree.

**Decision.**
- No module sits loose at a package root; every module lives in a subpackage named for what
  it is (`tests/agent_platform/test_package_layout.py`).
- The platform keeps its well-known subsystems at the top level and groups the small leaves:
  `core/` (primitives, persistence), `interface/` (tui, commands, presentation, host),
  `runtime/` (executor, scheduling, verification), `knowledge/` (artifacts, web, mcp).
  The evidence ledger lives with grounding (`hallucination_check/evidence/`); the Rill
  dashboards live with analytics (`analytics/dashboards/`).
- Runtime state is outside `src/`, one folder per package: `db/<package>/` for every
  database, `data/<package>/` for every other runtime file (memory, research runs, web cache,
  analytics Parquet, eval workspaces), `reports/<package>/` for eval reports. Every path is a
  config value; none is versioned.
- An agent's `domain/` never imports its `persistence/` (a Protocol port in `domain/ports.py`);
  no import cycles between an agent's subpackages.

**Consequences.** `config/paths.py` resolves every configured path against the repo root,
never the shell's working directory. Parked packages with no code (`multi_agent`, `sandbox`,
`prompt_compilation`) were deleted; they are listed in [PRD §6](PRD.md#6-parked-and-deferred).

**Supersedes.** The 2026-10-03 package-layout rule (extended 2026-10-12).

### P-04 — The agent registry: `make <target> <agent>`

**Context.** An unconditional default agent in the Makefile means running one agent's evals
while believing you ran the other's.

**Decision.** A registry at the top of the Makefile (`AGENTS := csa ra ops`, and per agent
`_PKG`, `_OVERLAY`, `_DATA`, `_RUNS`, `_EVALS`). Agent-scoped targets take the agent as their
second goal; the name alone sets `AGENTIC_OVERLAYS`. A missing, unknown or doubled name prints
usage and exits 2. `PROFILE=<name>` appends `src/<pkg>/config/profiles/<name>.yaml`. `verify`
imports each agent's composition root in its own process (two overlays in one process let the
first declaration win).

**Consequences.** Adding an agent is three registry lines. Old target names are kept as
aliases where muscle memory matters (`make data` → `make dummy-data`, `make queue` →
`make review`, `make llm-check` → `make model-check`).

**Supersedes.** ADR 0002.

### P-05 — Optional host hooks

**Context.** The research agent needs a greeting, a live side panel, a cancel key, and slash
commands that are conversational acts (they must reach the agent, not be answered by the shell).

**Decision.** Optional hooks, each its own `runtime_checkable` Protocol in
`interface/host/protocols.py`, checked with `isinstance`: `GreetingHost`, `PanelHost`,
`CancellableHost` (a priority key binding), `SubjectHost`, `MemoryHost`, `HousekeepingHost`,
`StatusHost`, `ActionHost`; and `ConsoleEffect.agent_turn`, which forwards a validated command
line to the agent as a turn.

**Alternatives rejected.** Widening `AgentHost` (every host implements hooks it does not need);
a second Textual app per agent (two shells, two sources of truth).

**Supersedes.** ADR 0004.

### P-06 — Episodes are configurations of the agents

**Context.** About 45 per-episode scripts and a separate pattern library drifted from the code
the viewer is meant to trust.

**Decision.** There are no per-episode scripts and no pattern library. An episode runs an agent
under the configuration it needs (features, profiles, `/harness`). [`patterns.md`](patterns.md)
records where each agent uses each pattern the series teaches.

**Consequences.** One code path per idea. `fixed_loop` has no implementation; a genuine fixed
loop is left for a later decision (pass^k reliability runs in the evals are the leading
candidate). PRD FR-58 and FR-59 are retired.

**Supersedes.** ADR 0006; the `design_patterns/` paragraph of ADR 0001; ADR 0005 (the one-off
baseline repair that preceded it, kept in git history).

---

## 2. The harness and its features

### P-07 — The harness is the set of declared features

**Context.** A component that consults configuration decides for every agent in the process,
which made two postures in one process impossible, and a "disabled" control that still ran and
reported finding nothing was indistinguishable on screen from a working one.

**Decision.**
- Nine features are toggleable: `deterministic_checks`, `prompt_injection`,
  `content_moderation`, `pii`, `intent_classification`, `hallucination_guard`, `gatekeeper`,
  `memory` (+ `working`, `episodic`, `semantic`, `procedural`), `context` (+ `rerank`,
  `failure_detection`). Sub-features are parent-gated.
- Each agent declares its posture in a `features:` block in its own config;
  `agent_platform` ships defaults and the `FeatureSet` mechanism (`config/features.py`) and
  never learns which agent is asking.
- The tree is read **once, in the composition root**, never inside a component. A withheld
  feature is a stage that is never built and never appears in the compiled graph.
- Everything else — the control loop and its budgets, resilience, observability, audit,
  logging, persistence, tool calling, schema validation — is always on: removing it produces a
  broken program, not a reckless agent.
- `/harness` is the only command that moves features (no args: the tree; `on`/`off`: all nine;
  `<path> on|off`: one). It is a hot reload — nothing read from disk, nothing remembered; a
  restart returns to the config. `/config` stays read-only.

**Consequences.** An agent declares its posture in full, even where it matches the platform
default, so the config reads as a statement of what the agent is. An agent cannot delete a
platform feature key, so it declares the ones it does not compose `false` (R-17).

**Supersedes.** CLAUDE.md "Feature toggles"; ra-0016 (in part).

### P-08 — `deterministic_checks` is the seam for an agent's business logic

**Decision.** The platform owns only `DeterministicChecker` — a request in, findings out.
`DeterministicCheckRequest` carries `subject_id`, an opaque string, so no agent is obliged to
have customers. Customer Support supplies `OrderChecks`; a new agent declares the feature and
injects its checker through `build_agent(deterministic_checks=...)`, changing nothing in the
platform. The stage runs after the security screens and before intent, so untrusted text is
vetted before anything reads a database on its behalf.

**Supersedes.** CLAUDE.md; the `basic_sanity` naming (renamed 2026-10-10).

---

## 3. Authority: Gatekeeper, Executor, standings

### P-09 — The Gatekeeper decides authority only

**Context.** One refund turn read the same order about fifteen times: the Gatekeeper re-ran the
refund policy, and the model was given a `check_refund_authority` tool that made it limit
itself, so turning the Gatekeeper off looked identical to leaving it on.

**Decision.**
- A fact about the **record** (the order exists, is the customer's, is refundable, the amount is
  its full total) belongs to the tool, checked once, with a conditional write.
- A fact about the **actor or the instruction** (offered this turn, RBAC, mandate, authority
  rank, the per-action limit) belongs to the Gatekeeper, which calls no model and reads no
  database per call.
- Every decision names a **standing**: VOID (denied; nothing queued), RATIFIABLE (escalated to a
  person; approval is RATIFIED), VOIDABLE (allowed, and undoable until it settles), VALID
  (allowed), UNGOVERNED (the Gatekeeper is withheld).
- Never log or show a control state that no code path produces or that cannot change a verdict.

**Consequences.** `check_refund_authority` and the agent-wide 24h amount window were removed; a
query-count test keeps the Gatekeeper database-free. Deferred: a live kill switch, `/policy`
commands, refund velocity, evidence-gated maturity levels.

**Supersedes.** The 2026-10-04 authority-only decision; the earlier `check_refund_authority`
tool.

### P-10 — PRD G1 is retired: the Gatekeeper may be withheld, the Executor may not

**Decision.** With `gatekeeper` withheld, a tool call executes with no authorization, budget or
surface policy — but still through the Executor (idempotency, deadline, lifecycle events). The
audit entry is still written on that path with standing UNGOVERNED, and no budget is committed.

**Consequences.** `/harness gatekeeper off` is a real demonstration: money moves. The Executor
is always on (`runtime/executor`); only an `ApprovedToolCall` reaches it.

**Supersedes.** PRD G1 ("every consequential action passes a single, always-invoked
Gatekeeper").

### P-11 — One tool path

**Context.** Tool calls reached execution by several routes, so a call the model proposed but
the platform rejected was never recorded, and tool validity could not be measured.

**Decision.** Every model-proposed call takes `tools/path.py` (`ToolPath`): registry (`unknown`
vs `invalid`) → loop control (`duplicate`) → Gatekeeper (`not_offered`, `refused`) or
UNGOVERNED → Executor (`executed`, then succeeded or failed). The stage it stopped at is
recorded as a `fact_tool_call`. MCP tools enter the same registry and take the same path.

**Supersedes.** ADR 0009 D9 (baseline refactor).

---

## 4. Memory, skills and human review

### P-12 — The agent profile and skills are procedural memory

**Context.** Each agent's prompt was a hard-coded string mixing identity, behaviour and every
procedure, carried on every turn. Procedural memory existed only on paper.

**Decision.**
- An agent's `AGENT.md` is its **profile** (Mission, Scope, Rules): always in the system prompt,
  versioned by content digest, never the authority for limits, permissions or record state.
- `skills/<name>/SKILL.md` files are **skills** (frontmatter `name`, `description`, `triggers`,
  `keywords`, `prefetch`, `tools`; body Preconditions, Procedure, Boundaries, Completion). They
  are synced into `SkillStore` at build time (a new version only when the text changes; one
  active version per name; every change audited).
- A turn's skills are selected **without a model call**: `always` skills, then skills whose
  trigger is among the turn's **signals** (opaque strings the agent derives, such as
  `intent:REQUEST_REFUND`), then — only if neither matched — BM25 over description and keywords
  above a floor; capped. Skills sit outside the context window: never evicted, not screened as
  data, not grounding evidence.
- **Skills must replace work, not add text**: only Procedure and Boundaries are sent
  (`skills.render.sections_sent`); `skills.max_prompt_chars` fails the build when the profile
  plus the largest skills outgrow the agent's budget; a skill's `prefetch:` reads run before the
  first model call; a turn is offered only the skill's `tools:`, by model name.
- An unplaced turn (`route:llm_fallback`) loads **no skill by default**. It gets one only by a
  lexical match over content words, or when it nearly reached a *read* intent; a skill that
  changes something is never loaded on an unplaced turn, and the turn is offered
  `intent_router.unknown_tools` at most. No skill never means every tool.

**Consequences.** A support refund turn went from 4 model calls to 2. The research agent has no
authored role skills (R-16). Owed: an end-to-end test of dream → approve → lexical selection.

**Supersedes.** ADR 0010 and its 2026-10-10 amendment; the 2026-10-12 unplaced-turn change.

### P-13 — Long-term memory is files; only a dream writes it

**Context.** Long-term memory was three storage engines and a fourth table for the dream inbox;
only the support agent could dream.

**Decision.**
- One directory per agent, `data/<package>/memory/{dream_events,episodic,semantic,procedural}/`.
  Records are numbered markdown files with YAML frontmatter under `.history/` (atomic writes,
  a cross-process lock); views (`semantic/<subject>/<key>.md`, `procedural/<name>/SKILL.md`,
  `MEMORY.md`) are regenerated; search is a derived FTS5 index, rebuilt when its fingerprint no
  longer matches. Paths cannot escape the directory.
- A completed turn writes a **dream event** (`TurnRecorder`); an agent may write others (a
  research run does). `make dream <agent>` / `/dream` runs a `Dream` with the agent's
  `MemoryProfile` — three consolidators, one per store. `MemoryProfile.default()` works out of
  the box: a template session summary, deterministic facts, and `RecurringSignalSkills`, which
  proposes a learned skill once a signal recurs in `skill_threshold` sessions.
- The model is never given a memory-write tool. The dream writes through the governed pipeline.

**Alternatives rejected.** mem0 (a remote LLM call decides every write; bypasses governance) and
Letta (owns the agent loop, conflicting with LangGraph); we keep mem0's reconcile vocabulary and
Letta's sleep-time consolidation. Anthropic's file-directory memory and SKILL.md progressive
disclosure were adopted, adding no dependency.

**Supersedes.** ADR 0012 (memory and dreams); the 2026-09-27 dream design.

### P-14 — One human-review queue, a comment on every decision

**Context.** A person decided things in four places (`/queue`, `/dlq`, `/skills approve`, and
nowhere at all for held facts); a comment was optional.

**Decision.** `platform_review_queue` in the agent's database holds every kind of item: the
platform's (`memory_fact`, `skill`, `quarantine`) and the agent's own (support: `refund`,
`cancel_and_refund`, `reply`). `ReviewService` refuses a decision without a comment
(`review.min_comment_chars`), reports an item outside the reviewer's scope as missing, runs the
kind's handler, then signs and audits. Three doors to one service: `/review`,
`make review <agent>`, and the console's Review screen (`ctrl+r`). A console scoped to a
verified subject shows that subject's items; before anyone is verified, only the agent's own.

**Consequences.** `/dlq`, `make dlq` and `/skills approve` were deleted. Escalation never pauses
the conversation: an item is queued and the person keeps chatting.

**Supersedes.** ADR 0012 (review queue); the 2026-10-03 one-HITL-queue decision.

### P-15 — PII never reaches a provider; tokens are reversible only on audit

**Decision.**
- Detection is Presidio over a local spaCy pipeline, with the configured regex patterns
  registered as recognizers (patterns-only fallback if Presidio cannot load). Entities are per
  agent: the platform default is contact and financial identifiers; Customer Support adds
  `PERSON` and `LOCATION`; the research agent keeps the default (names are research subjects).
- With `pii` declared, `PiiBoundary` wraps the agent's LLM clients: message text and tool-call
  arguments are tokenised on the way out and restored from that request's own map on the way
  back. Grounding, guardrails and tools work on real values; only the provider sees tokens.
- A token is `HMAC-SHA256(PII_VAULT_KEY, category|value)`; the value is AES-GCM encrypted in
  `platform_pii_vault` with a retention window. `make pii-reveal <agent> T=<token> WHY=<reason>`
  writes an audit entry before revealing. The audit trail and LangSmith exports hold tokens only.

**Supersedes.** The 2026-10-04 Presidio + vault decision; ADR 0012 (provider boundary).

---

## 5. Grounding and hallucination

### P-16 — The hallucination check runs in levels

**Decision.** One `hallucination_check` event per reply, carrying two levels, level 1 first.
- **Level 1 — token confidence** (platform, every agent): sentences scored by the completion's
  token log-probabilities (`llm.logprobs`). Confidence is not truth, so every claim still goes
  to level 2 — except two gates logprobs decide alone: a sentence whose content tokens average
  below `degenerate_mean_prob` is cut, and an id, amount or date written below `slot_min_prob`
  is cut unless the record confirmed it. A model that returns no logprobs makes level 1
  `skipped`.
- **Level 2 — the agent's grounding**: NLI entailment over tool results and documents
  (`ResponseGuardrail`, support and the research direct answer), the citation guard (research
  reports), or the figure check (ops).

**Supersedes.** ADR 0009 implementation notes (token-confidence step); the hallucination
check-levels decision.

### P-17 — Claims are grounded in one consistent world, cheapest evidence first

**Decision.**
- A reply is split into atomic claims and each claim is **classed**: `record` (a fact about this
  case), `rule` (how things work, and every promise), `conversational` (never scored). The class
  decides which tiers may ground it.
- Record claims: direct check (ids and amounts the record lacks, refused calls) → **fact
  ledger** (typed facts an agent puts in its evidence; comparisons computed, flags affirmed or
  denied) → state model (the refund state machine) → NLI over tool results → optional second
  opinion (confirm-only). Rule claims: direct check → fact ledger → policy → system prompt →
  memory (NLI, support only).
- Reads a later write or re-read has overtaken are dropped first. What stands is one consistent
  world: **any current entailment outvotes any contradiction**; a contradiction stands only when
  nothing current entails. Support must name what the claim names. Recency never votes.
- Premises are self-contained sentences that name their subject (config templates such as
  `order_sentences`). A contradicted claim whose fact the ledger read is *corrected*: its
  sentence is replaced by the fact's own sentence. An agent may set
  `unsupported_is_hallucination` (support does), making an undecided claim cut and queued.
- NLI and the reranker run locally (`local_models/`); the hosted encoders were removed.

**Supersedes.** The fact-ledger, consistent-world, local-NLI and order-record grounding
decisions (2026-09-26 → 2026-10-10).

### P-18 — Verified effects, not executor success, are evidence

**Decision.** After a side-effecting call, `runtime/verification` (`ReadBackVerifier`) re-reads
what should have changed through the agent's `EffectSource`. A call whose effect did not take
reaches grounding as `NOT EXECUTED`, so the direct check refutes any claim that it happened.
Deterministic checks are a pre-flight gate on the state before the action, not evidence.

---

## 6. Models, resilience and metering

### P-19 — Every model call goes through one chain builder

**Decision.** `llm/chain.py` (`ModelChainBuilder`) is the only place a model chain is composed:
one `ResilientCaller` per *model* (bulkhead → breaker → retry → timeout), hedged backups
(`resilience/hedge`, first success wins, a backup starts after `hedge_after_seconds`), recorded
attempts. Agent code never adds a timeout or retry around a model call; it configures the chain.
Every hosted model has a fallback, NVIDIA models only.

**Consequences.** A research run went from 367 s (failed) to 128 s (delivered) when the
duplicate agent-side retry loops were removed.

**Supersedes.** The 2026-10-02 platform model-resilience decision.

### P-20 — Agents write on small models; a different, stronger model judges

**Decision.** The line-up lives in config and changes there:
- Customer Support and Ops: the platform chain (a small fast model, backed by a larger one).
- Research writing roles: the agent's `x-default-chain` anchor, thinking off.
- The judge (`judge:` — eval graders and the research audit): the strongest model on the
  account, backed by one that is never the agents' primary, so no judge grades its own model.
- Hosted guards (jailbreak, content safety) fall back to chat models prompted as the same
  classifier (`security/guard_chain.py`); the request fails closed only when every one is
  unavailable. Embeddings have no cross-model fallback (vectors would not match the index).
- Card and event headings never name a model; the model goes in the body.

**Supersedes.** ra-0024; ADR 0013 (judge line-up); the 2026-10-03 and 2026-10-12 line-up
changes.

### P-21 — Queue time is not call time

**Context.** The rate limiter waits inside the HTTP transport, so a throttled call timed out in
the queue, was retried and queued again, and a queued model looked slow enough to start a hedge
on the same budget.

**Decision.** `resilience/admission.py` pauses a call's deadline while it waits for a rate-limit
slot and resumes it with the budget it had left; a hedge measures "slow" from admission.

**Supersedes.** ADR 0013 amendment (throttling).

### P-22 — One cross-process rate limit per provider

**Decision.** Every hosted request — chat, embeddings, guards, search — passes the sliding-window
limiter in `resilience/rate_limiter`, whose window lives in a shared SQLite store so every
process (console, evals, MCP server) draws on one budget. Buckets per host set; a per-bucket
monthly quota fails fast and is never retried; `Retry-After` on a 429 is honoured for every
process. Override per run: `RPM=<n>`.

**Supersedes.** ADR 0003 (quota extension); ra-0018 (free-tier guard).

### P-23 — Tokens are metered once, centrally

**Decision.** The only provider call (`OpenAICompatibleClient.complete`) reports usage to a
context-variable sink (`llm/metering.py`); a surface that needs a total binds `metered(...)`.
No agent or node counts tokens. One formatter (`StatusLineWidget.metrics()`) renders
"3m 27s · 9.7k tokens".

---

## 7. Analytics, MCP and dashboards

### P-24 — Analytics is a platform contract

**Context.** The first analytics build grew from what was easy to log: turn facts filled by
hand, mixed grains, only model calls recorded at a choke point, evals with their own schema.

**Decision.** Every agent gets the common metrics without writing analytics code.
- Facts are recorded only by platform choke points the agent cannot bypass: the turn bracket
  (`observability/units.py`), the task API, the model-chain builder, the tool path, the check
  recorder (`observability/checks.py`). Grains: task event, turn, model attempt, tool proposal,
  check. Common keys on every fact: `agent_id, session_id, task_id, turn_id, step, channel,
  recorded_at`.
- Task and Turn are both first-class. Outcomes are agent-defined in config, each flagged
  `success` and `terminal`; an undeclared outcome is a start-up error. Reserved outcomes:
  `unclassified`, `crashed`, `cancelled`, `abandoned`.
- Tier 1: a catalogue of common metrics, NULL (not 0) where a feature is withheld. Tier 2:
  agent-owned tables and views in the agent's own semantic layer, found by convention.
  Tier 3: eval results.
- An **analytics source** is a DuckDB plus its Parquet export of curated, documented views:
  `live`, or `dummy` (the generated demo world). Readers read only the Parquet;
  `SOURCE=dummy` points them at the demo. A running agent always records to `live`.

**Consequences.** A new agent appears on the Agent Platform dashboard by running turns. Releases
cannot be compared (no version column; an additive change if needed). Owed: eval facts in the
source (`eval_score`).

**Supersedes.** ADR 0007 (facts, mart); ADR 0009.

### P-25 — The analytics MCP server and the governed MCP client

**Decision.**
- **Server** (`knowledge/mcp/server/analytics`, stdio): read-only tools `list_metrics`,
  `compute_metric`, `list_views`, `describe_view`, `run_query`, and a `source://manifest`
  resource. Two walls: `QueryGuard` parses SQL with sqlglot and allows one SELECT over curated
  views (no table functions, no denied functions, an outer LIMIT); the database is in-memory,
  loaded from Parquet, external access disabled, under a deadline and a byte cap. A refusal is a
  tool error that tells the model what to change.
- **Client** (`knowledge/mcp/client`): a server's tools enter as ordinary `ToolDefinition`s and
  take the one tool path (P-11). The envelope (surface, side effects, permission) comes from
  `mcp.servers[].envelope` in the deployment's config — never from the server's own annotations.
  One session is owned by its own task (Active Object). Handshake, listing and every call are
  logged in full (`mcp.log`).

**Consequences.** Any MCP client (Claude Code, MCP Inspector) can launch the server:
`python -m agent_platform.knowledge.mcp.server.analytics`. Owed: an NL-to-SQL golden set scored
by execution accuracy; HTTP transport auth.

**Supersedes.** ADR 0008 (and its LLD "parked, interface-only" status).

### P-26 — Dashboards are generated, not hand-written

**Decision.** `python -m agent_platform.analytics dashboards` generates a Rill project from the
semantic layer: one **Agent Platform** dashboard (every agent side by side) and one per agent,
each with the same sections (Outcomes, Hallucination & grounding, Security, Tool use,
Reliability & speed, Cost, Models). Every measure is the metric catalogue's own expression over
raw rows, so it matches MCP `compute_metric` for any range. Nothing generated is versioned.

**Supersedes.** ADR 0007 addendum (two dashboards); ADR 0009 D3.

### P-27 — Traces: one LangSmith trace per turn, redacted

**Decision.** Each turn is one root run whose id is the turn's log `trace_id`; graph nodes,
model calls, executor tools and web calls nest under it. Payloads are converted to plain JSON
and PII-redacted before export, whatever the `pii` feature says. Tests delete the key so they
never export.

---

## 8. Evals and tests

### P-28 — Three graders, one shared judge, one make target

**Decision.**

| Grader | Kind | Grades | Gate |
|---|---|---|---|
| Oracle | code, zero model calls | the outcome in the database against ground truth | hard, 100% |
| DeepEval | model (the platform judge) | conversation quality, one anchored dimension per metric | soft |
| Promptfoo | adversarial | curated red-team attacks; assertions reuse the oracle's rules | hard, 100% |

- The oracle grades outcomes, not paths; the one tool rule is a forbidden tool that *executed*.
- Every section always runs; a failed gate is a result, not a process error. `make eval` exits
  non-zero only when a section could not run.
- One judge (`agent_platform/judge`): `LlmJudge` is a Template Method (`_answer` hook →
  `verdict`; any failure → `abstain`, never a silent zero). DeepEval's model adapter, Promptfoo's
  grader and the research audit all build on it.
- `make eval <agent> [oracle|deepeval|promptfoo] [full]` and `make intent-eval <agent>` are the
  only targets. A default run is the agent's **core** set (for support: 30 of 50 cases, 20 of 27
  attacks, 3 of 6 metrics — about 12 minutes); `full` runs everything. Generation is cached by
  (case hash, agent fingerprint); grading by (transcript hash, grader fingerprint).
- Trials: `evals.trials = k` reports pass@k and pass^k.
- One report per section per run in `reports/<package>/evals/`.

**Consequences.** The scoring cascade moved out of `evals/` into `agent_platform/scoring` (a
leaf package). The cascade's tier 5 (calibrated judge) still declines; the seam is
`scoring.CalibratedJudge`.

**Supersedes.** ADR 0013; ADR 0011 (bake-off target name); the 2026-09-27 framework.

### P-29 — Tests never call a provider

**Decision.** `tests/conftest.py::no_provider_calls` (session, autouse) deletes every configured
key variable and disables the hosted guards; models are scripted; databases are `:memory:`; NLI
uses the lexical model; LangSmith keys are removed. A test of a hosted guard re-enables it and
injects a fake client.

**Consequences.** The suite runs offline and deterministic. A test that only passes with live
embeddings is pinning a ranking; assert the contract instead.

### P-30 — One HALLUCINATION CHECK card format; each agent keeps its own check

**Decision.** `hallucination_check/card.py` owns the card's *format* only: `GroundingCard`
(result, tally, levels level 1 first, the token confidences, rule, `ClaimCheck` rows, removed
and corrected sentences, agent extras), `claim_tally`, `outcome_line`, `grounding_status` (what
was sent → completed / warning / blocked), `log_grounding` and `record_grounding_check`. No
check logic moved: the support agent's `ResponseGuardrail` decides exactly as before and now
calls the extracted tally, outcome and recording helpers (identical output). A contract test
validates its result as a `GroundingCard`, so the formats cannot drift. The TUI draws level 1
as the coloured confidence block and every later level as its row, for every agent.

**Consequences.** The support agent's card gains its "level 2" row; nothing it decides changes.
The stage key is defined once (`step_events.HALLUCINATION_CHECK`).

### P-31 — Every MCP protocol event is reported; the important ones are cards

**Decision.** `knowledge/mcp/client/events.py`: a session reports to an `McpEventSink` —
opening, handshake (`initialize` or `server/discover`, client info, capabilities, duration),
tools listed, a call that got no answer, every server notification (log messages at the
client's opted-in level, `tools/list_changed`, progress), clean close, session error.
`LoggedMcpEvents` writes each in full to `mcp.log`; `McpEvents` also draws MCP SERVER (the
handshake, or the connection failing), MCP CALL failed (no answer), MCP NOTIFICATION (tool list
changed; a server log at warning or above, `mcp.client.card_log_levels`) and MCP SESSION
(ended on an error). A changed tool list marks the source stale and it is re-listed next turn.
The analytics server logs each request and refusal, and tells the client a refusal (warning)
and a result to double-check (info) over `notifications/message`. An agent binds the sink in
its composition root; it draws no MCP card of its own.

The session's protocol profile is logged once, at the handshake (`MCP session profile`):
protocolVersion and how it was negotiated, serverInfo / clientInfo, both sides' capabilities
with their flags, every tool's inputSchema, outputSchema (the analytics tools return typed
models, so the SDK declares one), annotations (the server's claims) beside the local envelope
(what is enforced), and how calls run and fail (deadline, result limit, isError vs protocol
errors, progress, the `_meta` log-level opt-in). The MCP SERVER card shows the same, in
sections named by the protocol. It appears once per session, not per turn: the session stays
open, so there is no handshake on a later turn. Each `tools/call` logs only its arguments and
result. A notification the server sends while answering a call is logged under the trace of
the turn that made the call.

### P-32 — No JSON on a card

**Decision.** Tool arguments read as `key=value` (`text_tree.inline_arguments`), a tool's JSON
answer as a tree (`readable_lines`, `message_lines`), in the LLM REQUEST transcript, the CONTEXT
card (display lines in `PromptSection.shown`; `content` stays what is counted and sent) and
narrative cards. The same helpers serve every agent.

### P-33 — A tool call's TOOL RESULT is named by how far it got

**Decision.** `StepEvents.tool_outcome`: refused by the Gatekeeper (blocked), not run and the
stage that stopped it (warning), failed, or done — with an agent's own preview replacing the
raw output on success. Wording under `observability.step_events`.

---

## 9. Research agent

### R-01 — Two paths: a plain question is answered, `/research` researches

**Decision.** Chosen by the user, never guessed.
- **A plain message** is answered directly: a ReAct loop (`reason ⇄ act → ground`) over
  read-only tools (web search, page fetch, weather, clock), grounded against what the tools
  returned. A question that names no date gets the latest information.
- **`/research [question]`** starts a research scope: research kind → interview → rubric →
  `/approve` → autonomous run → independent audit → report.
- The refusal patterns apply on both paths before any model. With `direct_answer` withheld, a
  plain message starts a research scope.

**Supersedes.** ra-0020; ra-0017 (two modes); ra-0011 (the scopability gate).

### R-02 — Research kinds, each with its own definition of done

**Decision.** Five kinds: `decision_support`, `landscape_brief`, `fact_verification`,
`explainer`, `general`. A kind picks the interview checklist, the rubric template, the report
template and a stop-rule strategy class (`research/stop_rules/`). Cue rules, then an example
classifier (nearest labelled examples over a local embedder), then the LLM decide it.

**Supersedes.** ra-0011 (as amended by ra-0020).

### R-03 — Interview termination is hybrid and enforced in code

**Decision.** The interviewer may only *propose* done, and only when no required checklist item
is open; the user confirms. `/done` closes at any time — open items become **known unknowns**,
which the rubric must acknowledge and the report must list. At `scoping.max_turns` the agent
stops asking and offers `/waive <id|all>` or `/done`. An anti-sycophancy guard requires a
challenge or follow-up every K turns. A vague answer to a subject item leaves it open; after
repeated vague answers the item is `delegated` and the interviewer proposes values, marked
`derived`.

**Supersedes.** ra-0001; ra-0022 §6.

### R-04 — Acceptance criteria are generated deterministically, then edited

**Decision.** The rubric comes from `kind template + Brief` with no model call
(`rubric/generator.py`), then the user edits it (`/rubric add | drop | set`). Every user
criterion maps to at least one item; every known unknown to an "acknowledged and bounded" item;
every kind requires at least one cited finding, so no rubric passes vacuously. Cross-checks are
NLI checks over entailed claims from independent domains, and the item text says exactly that.

**Supersedes.** ra-0002; ra-0022 §5.

### R-05 — The contract is frozen and hashed at approval

**Decision.** At `/approve`, Brief + Rubric + Budget are frozen into a `Contract` whose canonical
hash is recorded in the audit ledger. The run graph receives only the contract; the hash is
re-verified at the end. A Brief may lower budgets, never raise them above the deployment ceiling.

**Supersedes.** ra-0014.

### R-06 — The plan is a DAG run as LangGraph supersteps

**Decision.** The plan is validated by `TaskGraph` (acyclic, closed, bounded, every node traced to
a rubric item). Each batch is a superstep: `batch_start` fans out one `Send` per ready node, each
`research_node` holds a slot of `AsyncDagScheduler`'s semaphore, `batch_done` joins. Budgets are
reserved synchronously before every await (`BudgetGuard.try_reserve`), so concurrent workers
cannot overshoot a cap. A checkpoint lands at every batch boundary.

**Supersedes.** ra-0005.

### R-07 — Three independent content-defence layers

**Decision.** **Structural** (outbound guard over every query and URL; source policy over every
hit), **classifier** (the platform's instruction/data separator over fetched text; a flagged page
is rejected whole), **quarantine** (only the extractor reads raw page text, inside a data
envelope, and it can return only verbatim spans). Each is toggleable; rejected pages are listed in
Coverage & Limits. The person's own message is screened separately by the platform
`SecuritySuite` (`prompt_injection`, `content_moderation`, `pii`).

**Supersedes.** ra-0007; ra-0016 (amended 2026-10-02).

### R-08 — The verbatim-span rule; provider summaries are never evidence

**Decision.** A ledger entry is admitted only if its span is an exact substring of the page text
this run fetched and its page hash matches (`VerbatimSpanValidator`). One span of one page is one
entry, whichever sub-question found it. A search provider's own prose (Tavily's `answer`) is kept
as `provider_summary` for display and never turned into a page, so it cannot be cited.

**Supersedes.** ra-0012, ra-0013.

### R-09 — Search behind an adapter and a router, inside free tiers

**Decision.** `knowledge/web`: `SearchProvider` adapters (DuckDuckGo, free and keyless; Tavily,
metered) behind a `SearchRouter` — DuckDuckGo first for general questions, Tavily first for
time-bounded ones and as the fallback. Tavily is capped by the shared monthly quota (P-22);
searches are cached on disk for 24 h and memoised within a run; a run spends a small search
budget. A spent quota falls back, and if nothing can search the run ends PARTIAL with the reason.

**Supersedes.** ra-0006; ra-0018.

### R-10 — Live by default; no synthetic world

**Decision.** The agent runs live (`web.mode: live`, NVIDIA models). The synthetic fixture world,
its reference model, the `demo` profile and the `model_backend` switch were deleted. Tests inject
scripted models through `build_agent(clients=...)`. A missing `nvidia_api_key` stops start-up
with a sentence; a missing `tavily_api_key` is a warning.

**Consequences.** **The research agent has no eval suite** until one is rebuilt over live or
recorded web data; `make eval ra` says so.

**Supersedes.** ra-0018, ra-0019; ra-0009 and ra-0015 (fixtures-only evals; fixture model
backend).

### R-11 — The independent audit sees three inputs and a different model

**Decision.** The audit runs on `IsolatedGraphRunner`; its entry point accepts only
`(contract, report, ledger_snapshot)` — any other key is refused. Checks run cheapest first:
deterministic → NLI → judge. The judge's chain excludes the writer's primary model
(`audit.independent_judge` holds start-up to that), and at run time an audit in which any
criterion was judged by the model that actually wrote the report is recorded as not independent.
The audit's judge subclasses the platform `LlmJudge`.

**Supersedes.** ra-0003; ra-0022 §3; ra-0024.

### R-12 — Budget exhaustion delivers PARTIAL, honestly

**Decision.** Exhaustion is never an error and never COMPLETE. The run synthesizes from the
ledger and delivers PARTIAL. **Coverage & Limits** is built in code from verdicts: each unmet
must-pass item, what it required, what was tried, and why the run stopped.

**Supersedes.** ra-0004.

### R-13 — Watch, stop, resume or abort — never steer

**Decision.** Once approved, the run is autonomous. The user watches (live stage view, side panel
with batches and budget meters) and may stop it (cancel key or `/cancel`); a stopped run keeps
its checkpoint and its state is printed, then `/resume` continues it or `/abort` ends it (the
run's record is kept). There is no mid-run steering and no contract edit after `/approve`. These
run controls are the platform's (`durability/controls.py`), shared by every durable agent; a
saga compensates on stop.

**Supersedes.** ra-0010; the 2026-10-04 run-controls decision.

### R-14 — Reflection on the evidence and on the report

**Decision.** Audit failures are routed by fault. A failure of a check in
`reflection.writing_checks` is a writing fault sent to `revise` (Producer-Critic: the writer gets
its previous sections and the audit's findings, then the revision goes through the same citation
guard and audit), bounded by `max_revisions` and stopped on a plateau. Any other failure is an
evidence gap sent to one bounded gap-fill round. Evidence goes first.

**Supersedes.** ra-0022 §§1–2.

### R-15 — What a run learned reaches the next run through dreams

**Decision.** A finished run writes one `run` dream event whose signals are what the critics found
(`criterion_failed`, `source_rejected`, `budget_stop`), scoped `archetype:<kind>`.
`make dream ra` consolidates them through `research_memory_profile()`: an episodic run summary,
rejected domains as semantic facts, and — once a finding recurs in `skill_threshold` runs of one
kind — a proposed learned skill a curator approves in `/review`. At plan, learned skills for the
kind are selected without a model and handed to the planner, reflector and reviser.

**Consequences.** One bad run no longer changes the next one's prompt; a lesson needs a pattern.
The LESSONS card and the `reflexion` feature were deleted.

**Supersedes.** ra-0023; ra-0022 §4 (Reflexion).

### R-16 — The profile goes on the direct answer only; no authored role skills

**Decision.** Each run role's own instructions are its procedure, so the agent has no authored
role skills. The profile (`prompts/AGENT.md`) is placed on the direct answer only
(`research_agent.instructions.profile_roles: [answerer]`). `memory.procedural` is on for learned
skills only (R-15).

**Supersedes.** ra-0021.

### R-17 — Platform features the research agent does not compose

**Decision.** An overlay cannot delete platform feature keys, so the agent declares
`deterministic_checks`, `intent_classification`, `hallucination_guard` and `gatekeeper` false —
it has no business checker, no intent router, its own citation guard, and no side-effecting tool
for a Gatekeeper to authorize. `prompt_injection`, `content_moderation`, `pii`, `memory` and
`context` are composed.

**Supersedes.** ra-0016.

### R-18 — Three frozen artifacts per run

**Decision.** Brief, Rubric and Report — each with a Markdown rendering — are frozen, hashed
artifacts in `DirectoryArtifactStore` (`data/research_agent/runs/`). One index row per run goes to
episodic memory. The evidence ledger is discarded: the report's citations carry URL, span, page
hash and retrieval time. `make runs ra` lists runs and re-verifies every hash.

**Consequences.** Runs archived before a schema change may fail to load (`make runs ra` raises on
an old `archetype: qa`); they are kept as history.

**Supersedes.** ra-0008.

---

## 10. Customer-support agent

### C-01 — Business rules live in code, not in the prompt

**Decision.** Refund and cancellation rules are domain code (`domain/refund_policy.py`,
`domain/cancellation_policy.py`), asked by the tools; knowledge-base articles explain and never
decide. A refund applies only to a settled order (delivered, or cancelled before delivery), is
always the order's full total, and is discussed only when the customer raises it. Dummy data
obeys the same rules.

### C-02 — The refund limit lives in the database and is enforced by the Gatekeeper

**Decision.** The per-refund limit's only source is the `platform_agent_budgets` row
(`make refund-limit <amount>`). The Gatekeeper (`CustomerSupportBusinessRules`) escalates a refund
above it (RATIFIABLE → review queue); the model's money-moving tools are `process_refund` and
`cancel_order` (there is no `queue_for_approval`). The limit is rendered into the prompt as one
sentence from the database (so a gatekeeper-off demo shows a persuasive prompt talking the model
past it), and the oracle checks it is never disclosed in a reply.

**Supersedes.** The 2026-09-27 and 2026-10-10 refund-limit decisions.

### C-03 — A refund is issued, then settles

**Decision.** `RefundSaga` marks the refund **issued** (one conditional update carrying the state
the caller saw, the full total and a refundable order state), then records the payment; each step
has a compensation and the saga is idempotent. An issued refund is VOIDABLE until its window
closes: `/void` reverses it and releases its idempotency records. When the window closes the
console's housekeeping settles it (completed), and only then is the customer notified.
`CancellationSaga` cancels a pending order first, then runs the refund steps; above the limit it
cancels now and refers only the refund.

### C-04 — Authentication is the hard front door

**Decision.** A customer is verified by a seven-character user code before anything acts on their
behalf. Identity is a context variable bound by authentication, never by a tool argument. An
unauthenticated request never reaches screening, intent, retrieval or the model.

### C-05 — Grounding reads the order under discussion, fresh

**Decision.** The grounding evidence includes a fresh, customer-scoped database read of the order
under discussion (`order_record`), never text parsed from the prompt window. Write intents carry
one order only. The deterministic draft check (`GroundedCritic`) is a pre-filter for the guard,
shown as one row on the HALLUCINATION CHECK card.

### C-06 — Intent: Laya by default, the classifier a config switch

**Context.** The fine-tuned BERT classifier had no label for two of the seven intents, folded 21 of
28 labels into UNKNOWN, and was confidently wrong. On a 50-message bake-off Laya reached every
intent and was right 95% of the time it decided at 0.6 (BERT: 28% alone).

**Decision.** Phrase rules first (`intent_router.phrase_rules`), then the local classifier at its
own threshold: Laya (`laya_local`, asked which intent description fits) by default, BERT
(`huggingface_local`) by profile `bert_intent`. Each provider's threshold and label mapping sit
with it in `intent_router.classifiers.<provider>`. Reads route deterministically
(`deterministic_read`) with no model call. The platform's `LayaIntentClassifier` takes its labels
from the agent at build time.

**Consequences.** About 185 ms more per unsettled turn on CPU; an intent is added by writing its
description. Open: review the 50 labels; refit Laya's temperature on real traffic.

**Supersedes.** ADR 0011.

### C-07 — The agent database is built from its seed

**Decision.** `db/customer_support_agent/agent.db` is gitignored and built by
`make dummy-data csa` from `generator.yaml` and the seed SQL. Start-up refuses an empty database
(`require_seeded`) rather than answer every user code as unknown.

---

## 11. Ops agent

### O-01 — The ops agent's only tools are the analytics MCP server's

**Decision.** A LangGraph ReAct loop (`propose ⇄ execute → ground`) whose only tools come from the
platform's analytics MCP server through the governed client: each call is validated, authorized
by the Gatekeeper against `ops_agent.policy`, run by the Executor and audited. Its
`hallucination_guard` is a figures check — every number in the answer must appear in a query
result — after level-1 token confidence. The reply shows the SQL it ran and the rows it rests on.
It declares `prompt_injection`, `content_moderation` (added with O-03), `gatekeeper`,
`hallucination_guard` and procedural memory (one
`always` skill, `analytics-investigation`); it keeps no episodic memory.

**Supersedes.** ADR 0008 (ops agent).

### O-02 — The ops check is the support agent's, figures in place of NLI

**Decision.** `OpsGrounding`: level 1 token confidence, level 2 every figure looked for in the
query rows (`FigureCheck`, by sentence and row), then the platform's confidence `gate` exactly as
the support agent applies it — a degenerate sentence, or one holding a figure written below
`slot_min_prob` that no row holds, is cut (TRIMMED; nothing left: the configured withheld
reply, BLOCKED). A figure a row holds is confirmed whatever its probability. A confidently
written figure no row holds is kept and named as unverified in the reply (FLAGGED, a warning).
The card is P-30's, each figure's step quoting the query and row that hold it.

**Supersedes.** O-01's "level 1 is shown, not used".

### O-03 — The SQL is reviewed against the question before the answer is grounded

**Decision.** Deterministic first: DuckDB binder, catalog, parser and conversion errors come back
as a guided rejection naming the view's real columns and the closest match; a result with no
rows or cut rows carries a note to check. Then a Reflection critic (`SqlReviewer`): once the
model answers, one model call reads the question, the model-written SQL (`run_query` only — a
catalogue metric runs the catalogue's formula), a few rows and the draft, and says accept or
revise. Revise sends the finding back to the model, at most `sql_review.max_reflections` times a
turn; the governor's budgets count the call. `compute_metric` answers are reviewed too: the
formula is the catalogue's, but the metric, agent and period are the model's choice.

**Not reviewed.** A turn the governor stopped (a budget or the step cap) goes straight to
grounding, as does an answer that rests on no query.

**Fewer wandering calls.** An `agent_id` the source holds no data for is refused with the ids
it does hold (`list_metrics` also lists them), instead of returning 0 rows the model then
explores; an empty result says only that, with the right filters, there is no data. The token
budget is sized so the step cap, not the token cap, ends a long investigation. A critic that fails or answers out of form
accepts, and its REFLECT step card says the answer went on unreviewed. Built with
`hallucination_guard`; no new feature.

**Why both.** The figures check proves a number came from a query, not that the query answered
the question; only reading the SQL against the question catches the wrong filter or aggregate.

---

## Appendix A — superseded and retired decisions

| Original | What it said | Now |
|---|---|---|
| ADR 0001 (part) | Keep `design_patterns/` as the teaching layer; `make patterns` | Retired by P-06 |
| ADR 0002 (part) | `make data <agent>` records RA's fixture world; `ABLATION=`; `eval-diff` | Retired with the fixture world and the eval rework (R-10, P-28) |
| ADR 0005 | One-off baseline repair before the research agent | Historical; kept in git history |
| ADR 0007 | `platform_llm_calls` / `platform_tool_calls` / `platform_turn_outcomes`, the `turns` / `events` views, two dashboards | Replaced by P-24, P-26 (`make analytics-drop-legacy` drops them) |
| ADR 0008 (naming) | "mart", `mart://manifest`, `make analytics-mcp`, `agent_platform.analytics.mcp_server` | "source", `source://manifest`, `agent_platform.knowledge.mcp.server.analytics` |
| ADR 0010 (part) | "Charter", "procedure", `ProceduralStore`, `/skills approve`, three RA role skills | "Profile", "skill", `SkillStore`, `/review`, no RA role skills (P-12, R-16) |
| `check_refund_authority` tool | The model checks its own refund authority | Removed; the Gatekeeper decides (P-09) |
| Agent-wide 24h refund window | Cumulative amount window for CSA | Not enforced (`gatekeeper.ledger.enabled: false`) |
| `hitl_queue`, `/queue`, `/dlq`, `make dlq` | Separate escalation stores and commands | One review queue (P-14); `make queue` kept as an alias of `make review` |
| Hosted NLI and reranker | Encoders behind a hosted API | Local models (P-17) |
| ra-0009 | Fixtures-only, deterministic evals | Superseded by ra-0019 → R-10 (no RA suite) |
| ra-0015 | `model_backend: fixture | routed` | Deleted (R-10) |
| ra-0017 | Two modes, Q&A and Research, chosen by a gate | Superseded by R-01 |
| ra-0022 §4 | Lessons written straight to episodic memory; LESSONS card | Superseded by R-15 |
| PRD G1 | Every action passes an always-invoked Gatekeeper | Retired by P-10 |
| `control_loop` package (parked) | FSM + `LoopController` | Built as `loop_control` (`LoopGovernor`) |
| `episodes/`, `make episode` | Per-episode scripts | Retired by P-06 |
