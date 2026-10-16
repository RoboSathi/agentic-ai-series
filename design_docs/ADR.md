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
| [Evals and tests](#8-evals-and-tests) | P-28 – P-35 |
| [Research agent](#9-research-agent) | R-01 – R-20 |
| [Customer-support agent](#10-customer-support-agent) | C-01 – C-08 |
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
`AsyncDagScheduler`), `loop_control/budget.py` (`BudgetGuard`),
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
  `runtime/` (executor, scheduling, verification), `knowledge/` (web, mcp).
  The evidence ledger lives with grounding (`hallucination_check/evidence/`); the Rill
  dashboards live with analytics (`analytics/dashboards/`).
- Runtime state is outside `src/`, one folder per package: `db/<package>/` for every
  database, `data/<package>/` for every other runtime file (memory, web cache,
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

**Amended by** P-40: values are tokenised at the source; the harness itself works on tokens.

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

### P-34 — The CONTEXT card shows the tool schemas; crowding is judged on large requests only

**Context.** The card's tree said `Tool schemas · 566 tokens · 5 tool(s)`, but the schemas
were not on the card: every other section had a full-text body, and the tools had none. The
confusion check then flagged a 1.2k-token research answer as "too many tool choices: 5 tools;
their schemas are 47% of the request". The limit is 12 tools. It was the share rule that fired,
and the share is meaningless on a lean prompt: 566 tokens crowd nothing out of a 131k window.

**Decision.**
- `PromptManifest.tool_specs` carries the schemas as sent. The card's tree gives the tools line a
  share like every section, and a `Tool schemas` body shows every tool in full as a tree (name,
  description, parameters).
- The confusion check is two named rules:
  - `confusion_tools` fires on the count only (`max_tools`).
  - `confusion_schema_share` fires on the share (`max_tool_schema_share`), and only for a
    request of at least `min_request_tokens_for_share` tokens.
  Each finding names the rule that tripped and its limit.

**Consequences.** A lean agent prompt is never flagged for its own leanness. Crowding is still
caught where it can hurt: many tools, or schemas dominating a large request.

### P-35 — A claim is projected once; a card renders the check's steps, never re-decides

**Context.** Three layers each decided again what settled a claim. The guardrail built the
card's dict (`_claim_check`, with its own `_rank`). The TUI then ranked that dict's evidence once
more (`_strength`, `_deciding_evidence`, `_decided`), matching string labels, and imported
`CONFIRM_TIER` from the resolver. `readings.py` had a third ranking (`_deciding`). The NLI label
mapping `{unknown: neutral}` existed four times, and the platform guardrail hard-coded the CSA's
"customer message" tier name.

**Decision.**
- `hallucination_check/claim_view.py` is the one projection from a verdict to a claim on the
  card and in the log (verdict word, method, basis, per-model scores, evidence, steps).
- `readings.deciding` is the one ranking of a claim's evidence; `readings.label_name` is the
  one label mapping.
- Tier names, label names and provenance wording live in `hallucination_check.card`; an agent
  names its own tiers in its overlay.
- The TUI renders a claim's `readings` steps. A claim no step read shows the check's own
  reason. The widget no longer ranks evidence.

**Consequences.** `guardrails/response.py` holds only enforcement (validate, check, repair,
trim). Card tests now go through the real projection rather than hand-built dicts.

### P-36 — A poisoned line is struck at its source; a question, a scope or an offer asserts nothing

**Context.** The context health check cut the model's own false claim ("Order 10633 has been
cancelled and refunded.") from one turn's window only. Working memory still held it, so the
next turn gathered it again, and the check found it, cut it and reported "REMOVED" on every
turn. Separately, the hallucination check withheld correct out-of-scope replies. "How can I
help you with order 10633?" names an order, so it was classed as a record claim and refuted
by the order's state. "I can only help with your orders" and "I can check its status for you"
were classed as rules and NLI-scored against unrelated prompt or KB lines.

**Decision.**
- A source that holds its own text implements `RetractableSource.retract`, and
  `FiveStagePipeline.retract` fans the call out to such sources. After mitigation,
  `ContextWindow` strikes the cut lines of findings in
  `context.failure_detection.retract_modes` (`[poisoning]`) from their source.
  `WorkingMemory.strike` removes the sentence from the conversation and drops any entry it
  empties. The finding is marked `struck`, and the card says so (`struck_line`).
- A rot reduction or a clash keep is a choice about one window and is never written back.
- A claim cut from a question that states no outcome is conversational, and this is checked
  before the record check. The agent stating its own scope is a conversational opener, and
  the offer verbs include "check" and "look into/up". All three are configuration
  (`hallucination_check`).

**Consequences.** A poisoning finding appears on the turn it is caught and not again. "Did
you know order 10633 was refunded?" still states an outcome and stays a record claim.

---

### P-37 — The memory view shows what dreams consolidated; dream events are a view of their own

**Context.** `make show-memory`, `/memory` and `make dream SHOW=1` ended with the dream inbox,
so raw events waiting to be consolidated were read alongside what memory actually holds.

**Decision.** `snapshot_tables` returns episodic, semantic and procedural tables only. The dream
events are `inbox_table`, reached by `make show-memory dream <agent>` and `/memory dream`. `dream`
is a view keyword of `show-memory` in the Makefile, never the `dream` target, so naming it there
cannot consolidate anything.

**Consequences.** One shared builder keeps the terminal and the console in agreement.

### P-38 — The judge's own model is never an agent's; its backup may be

**Context.** The agents moved to super-120b, which was also the judge's backup, so a stalled
ultra call was graded by the agent's own model.

**Decision.** The judge (`judge:`, and the research agent's `judge` role) is ultra-550b. When
ultra does not answer, super-120b grades instead: for a demo, a grade always coming back
matters more than an independent one (Karan, 2026-10-15). The agents' chain (`llm.chain`) is
super-120b with no fallback, so an agent never runs on the judge's model.

**Consequences.** A grade made while ultra is down is the agent's model grading itself; the
call's model is logged, so such grades can be told apart. When super-120b is unavailable a CSA
or Ops call fails after its retries. Amends P-19 ("every hosted model has a fallback") and P-20.

### P-39 — The platform lays every request out for the provider's prefix cache; every call says whether it hit

**Context.** A provider's prefix cache serves the longest byte-identical opening it has
already seen, and cached input costs less and returns sooner. The `Cache` strategy and the
manifest's stable prefix existed, but each agent decided its own order, so the variable part
got into the prefix: CSA put its per-intent skills ahead of the policy and limit; tool schemas
went out in the selector's ranking order; a history-carrying layout put the per-turn window
ahead of the history; and a research role's static schema came after its payload. In the logs
on 2026-10-15, ~313 requests shared ~40 prefix signatures, but only 3 of ~840 responses
reported `cached_tokens > 0`. Hosted NIM is load-balanced with no prefix affinity.

**Decision.**
- `llm/prompt.py` `Volatility` (`FIXED` < `SESSION` < `TURN`) is on every section and
  `PromptPart`. `PromptBuilder.build` orders the system sections by it, so an agent cannot add
  them in a cache-losing order. With history, the `TURN` sections move to their own message
  between the history and the question (`context.prompt.turn_context`: role `user`, since not
  every chat template accepts a later system message). Without history they close the system
  message. Placement is overridden through `_order_system`.
- Skills are `TURN` (chosen per turn). The window's `CacheStrategy` part is `SESSION`.
- `ToolRegistry.to_openai_tools` returns schemas in registration order, whatever order it is
  asked in. The tool schemas are part of `prefix_signature` and `prefix_tokens`.
- A research role's answer schema is in its system message; the user message is the payload.
- Every LLM call reports `cached_input_tokens` and `cache_hit` as the last fields of its
  response card, and in the adapter's log with the provider's raw `usage`.
  `TokenUsage.cache_reported` tells "the provider did not say" apart from a miss. The central
  meter passes the whole `TokenUsage`, and the turn summary's `prompt_cache` line sums it.

**Consequences.** The layout gain shows on a provider that caches (llama.cpp locally); on
hosted NIM, `cache_hit` mostly reads false, and the reason is now visible instead of guessed.
A research role's request text changed (schema moved), so its prompt cache and any recorded
fixtures that match on it start over.


### P-40 — PII is tokenised at the source; a value comes back only for a tool or a person

**Context.** A turn summary showed `verified_customer: Pythagoras (account #19)` with `pii`
on. The redactor had run, but spaCy NER finds no PERSON in that fragment (tested: none even at
threshold 0); it does find "Ada Lovelace (account #3)". Redaction was also applied one call
site at a time (memory notes, tool cards, the summary, the dream inbox, LangSmith, the review
queue), so file logs, checkpoints and any new card were one forgotten call from a raw value.

**Decision.** (Karan, 2026-10-15/16.)
- **Known values.** A value the harness read from its own records is registered
  (`security/pii/known_values.py`) and found by exact, word-bounded match ahead of the model.
  An agent declares which columns of its records hold personal data
  (`security.pii.known_columns`; CSA: `users.name/email/phone`), and the database pool reports
  every row it reads, so those values are registered before the read is even logged. A value the model finds at the source is
  registered too. The registry is in process memory only. NER stays Presidio with spaCy lg.
- **At the source** (`security/pii/source.py`, read hot through `security/pii/active.py`):
  - Each agent's message is admitted (tokenised) at the start of `handle`, before the trace,
    the checkpoint or the run registry holds it. The screening stage draws the PII card from
    that admission.
  - The Executor tokenises a tool's result and restores its arguments.
  - `AdmittedChecker` tokenises a deterministic check's facts.
  - The verification read-back is built from tokens.
- **Values come back** in only four places: a tool's arguments, the reply shown to the person
  (the TUI; an eval grades the same), a reviewer-approved reply, and the audited
  `make pii-reveal`. Restoring reads an in-memory token → value map, so it needs no vault key.
- **Every writer scrubs.** Log records and status events (every card and the turn summary)
  pass the installed scrub (`core/primitives/scrubbing.py`): known values plus patterns, no
  model. `PiiBoundary` stays at the provider.
- `/harness pii off` makes every hook the identity.

**Supersedes.** P-15's "grounding, guardrails and tools work on real values" — they now work
on tokens. HMAC tokens are deterministic, so evidence and reply still agree.

**Consequences.** Logs, checkpoints, memory, audit and cards hold tokens; a value is
recovered only through `make pii-reveal`. A PERSON false positive in a product name becomes a
token and is restored for the customer. Each written string costs a regex pass.
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
searches are memoised within a run (no disk cache since R-21); a run spends a small search
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

### R-18 — Runs are not kept; every question is researched fresh

**Context.** Each finished run used to be archived: Brief, Rubric and Report as frozen, hashed
files, an episodic index row, and optionally a semantic pointer. A later question then had up to
three of those runs offered back as "Earlier research" pointers in its context. Web research goes
stale, and a pointer to last week's report invites the model to lean on it instead of searching.

**Decision.** Nothing of a run outlives it.
- The run graph ends `finalize → close → END`. `close` leaves the critics' findings in the dream
  inbox (they teach skills, R-15; they hold no evidence) and discards the evidence ledger.
- No artifacts, no index row, no semantic pointer, and no earlier-run context source.
- The direct answer's context window has working memory as its only source.
- `make runs ra` is gone, and so is the platform's `knowledge/artifacts` store, which had no
  other user.
- The contract is still frozen and hashed at approval and re-checked at the end of the run
  (R-05). The audit trail records the run.

**Consequences.**
- Every question pays for its own searches.
- A delivered report lives in the conversation and the audit trail only.
- LangGraph checkpoints remain, because Stop/Resume of a run in flight needs them (R-13).

**Supersedes.** The earlier R-18 (three frozen artifacts per run), ra-0008.

### R-19 — The report chooses its own outline

**Context.** The synthesizer was given fixed headings per research kind (explainer: Summary,
Concepts, Examples) and told to use exactly those. Most questions do not fit a fixed form, and
a section with nothing true to say invites filler.

**Decision.**
- `report_templates.yaml` gives each kind a `shape`, a one-paragraph hint (decision support:
  open with the recommendation and why).
- The writer chooses headings specific to the question and evidence. It answers the question
  in the first section, honours the brief's `deliverable_format`, and writes at most the depth's
  `report_max_sections`.
- Code holds the outline to its limits after the draft (`synthesis/outline.py`):
  - empty sections and reserved headings (the sections the run appends) are dropped;
  - a heading used twice becomes one section;
  - sections past the cap fold into the last kept section under their own heading, so no cited
    sentence is lost.
- A revision keeps the previous headings unless a finding asks otherwise.

**Consequences.**
- Reports differ in shape from run to run. Readers, the citation guard and the audit already
  read whatever sections exist.
- An empty ledger yields no written sections at all; Coverage & Limits says why.

### R-20 — The run's loop policy is one pure unit

**Context.** `run_nodes.py` decided when a run stops, where it goes after the audit and how it
ends inside its node methods, against live resources. The card and the edge each re-ran those
private methods, and the only way to test a branch was to build a graph.

**Decision.** `research/loop_policy.py`: `RunLoopPolicy.after_reflect`, `.after_audit` and
`.final_status` take a `LoopSnapshot` (counters spent, budget stops, rounds left, cancelled,
unreadable-only) and return a decision with its reason and cause. Nodes build the snapshot;
edges and LOOP CONTROL cards ask the same policy. The rescue and report-check card builders
moved to `research/cards.py`.

**Consequences.** Every branch is unit-tested without a graph (`test_loop_policy.py`). The
platform `LoopGovernor` still does not govern the research run loop; that remains the known
follow-up.

### R-21 — No on-disk search cache

**Context.** `SearchCache` wrote one file per live search to `data/agent_platform/web_cache/` and
honoured its 24 h TTL only on read; nothing deleted an expired entry, so the folder grew with
every run. R-18 already researches every question fresh, so answers reused across runs worked
against that rule.

**Decision.** `knowledge/web/cache.py` and `web.search.cache_ttl_hours` / `cache_dir` are
deleted. The gateway memoises searches (and fetches) for its own lifetime only, so a repeat
within a run still costs no provider call and still reads "cached" on the card.

**Consequences.** A question asked again in a later run spends a search again; the shared
monthly quota (P-22) and the per-run search budget remain the spend controls. Runtime leftovers
from retired features were removed with it: the Kelvari fixture set (`web_fixtures/`, R-10) and
archived runs (`data/research_agent/runs/`, R-18). Record / replay stays as a generic mechanism.

**Supersedes.** The disk-cache clause of R-09.

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

**Amended 2026-10-15: one sentence per Gatekeeper posture.** The composition root picks the
sentence by whether the Gatekeeper was built (`refund_limit_notice.governed` / `.ungoverned`). The
grounding evidence uses the same one, and `/harness` rebuilds the agent, so the choice follows the toggle.
With the Gatekeeper on, the model calls `process_refund` with the full total whatever it is, and an
over-limit call is escalated to the review queue. With it off, the sentence forbids that call and
points the customer to support. It is the only guard left, so a plain request is declined and
a persuasive one can talk the model past it. Before this, the governed sentence was used in both
postures, so with the harness off it invited the model to refund above the limit and there was
nothing to jailbreak. The ungoverned sentence forbids the call "until a person has reviewed it",
not unconditionally. An unconditional "never" gave the model nothing to argue with (super-120b quoted it
and declined a staging-QA prompt). A condition is what a real prompt rule carries, and a forged
"the review is done" talks the model past it. That is the failure the Gatekeeper exists for.

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

### C-08 — Grounding evidence and turn memory are services, not mixin methods

**Context.** `NodeSupport` (the base of every node mixin) carried ~230 lines that assembled a
reply's grounding evidence. The agent class carried the writes a finished turn makes to memory.
Each changes for its own reason, and as mixin methods every node could reach them and every
collaborator they used.

**Decision.**
- `grounding/evidence.py`: `GroundingEvidence.for_turn(TurnRecord)` covers the order record,
  tool results and refusals, the policy and prompt rules, the articles, the typed facts and the
  customer's words.
- `memory/turn_memory.py`: `TurnMemory.complete(FinishedTurn)` covers working memory, the dream
  inbox, and the claim offered to semantic memory.
- Both read plain records, never `AgentState`, so neither imports the graph. The agent builds
  both; `NodeSupport` keeps one-line delegates.

**Consequences.** The mixin structure and the agent's constructor are unchanged; changing them
would cost more than it buys.

**Known constraint, not changed.** `security/pii/active.py` reads the process-wide published
posture (`active_features()`) on every call, because review surfaces built outside a composition
root must follow `/harness` hot. That is correct while a host runs one agent. Two agents with
different `pii` postures in one process would need the host to own a posture per agent and hand
it to those surfaces.

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

### O-03 — REFLECT: the data is checked after the query runs and before the model answers

**Decision.** Deterministic first: the server's walls run before any SQL executes (QueryGuard;
DuckDB binder, catalog, parser and conversion errors returned as a guided rejection naming the
view's real columns and the closest match; an unknown agent id refused with the known ones),
and a result with no rows or cut rows carries a note. Then REFLECT, a node between
`execute` and the model's next step (`propose → execute → review → propose`), after every
step that ran a data query:
- **Model-written SQL (`run_query`):** a Reflection critic (`SqlReviewer`) reads the
  question, the SQL and a few of the rows it returned, and says accept or revise. Revise
  appends the finding, and the model fixes the query before it writes anything; accept lets
  it answer from those rows. The answer is written once, from reviewed data. At most
  `sql_review.max_reviews` reviews a turn; past the cap the model goes on, and the card says
  the SQL was not reviewed. Several queries in one step share one review. The call counts
  against the governor's budgets; a critic that fails or answers out of form accepts.
- **Catalogue calls only (`compute_metric`):** no model call. The model wrote no SQL; the
  server ran the catalogue's formula with arguments it checks deterministically. The REFLECT
  card is decided in code (`StepCard.decided_in_code`) and says exactly that.

**Why after execution, not before.** The store is a read-only sealed copy, so running a query
harms nothing, and the rows are evidence: only after a query runs can a reviewer see an empty
result or a wrong id. What must never run is stopped before execution by the server's walls,
not by a model.

**Why both.** The figures check proves a number came from a query, not that the query answered
the question; only reading the SQL against the question catches the wrong filter or aggregate.

**Fewer wandering calls.** `list_metrics` lists the agent ids; an empty result says only that,
with the right filters, there is no data. The token budget is sized so the step cap, not the
token cap, ends a long investigation.

### O-04 — Free SQL is its own permission; tool results are data

**Decision.** A tool may carry its own envelope (`mcp.servers[].tools.<name>`) over its
server's. `run_query` requires `query_analytics`, which the catalogue tools do not, so a
policy that grants only `read_analytics` has free SQL refused by the Gatekeeper. Every tool
result reaches the model inside the platform's data delimiters (`InstructionDataSeparator`,
with `prompt_injection`); an injection phrasing found in a result is logged. MCP in front of
the store is where these controls live for any client: credentials and the connection stay
server-side, only curated views are visible, SQL is parsed and allow-listed, results are
bounded, and every call is governed and audited locally.

---

### O-05 — A spent budget and a cut-off answer are said as what they are

**Decision.**
- **The loop never charges a call it stopped.** `LoopGovernor.before_action` charges the
  tool-call budget only for a call it lets run; a call stopped because the budget is spent is
  neither run nor charged, so the counter never reads past its cap. This is a platform fix,
  for every agent.
- **Who stopped a call is named.** The TOOL RESULT card says the loop layer
  (`not run · tool_budget · tool_calls: 16 spent of 16`) or the Gatekeeper
  (`refused by the Gatekeeper · <its reason>`), with the reason in the heading. The model
  reads "the tool-call budget is spent", never a Gatekeeper refusal.
- **After the budget is spent, no tools are offered.** The model is told to answer from the
  results it has, and to say which figures it could not fetch.
- **A cut-off answer is never sent.** The model reasons before it answers, inside the same
  cap. An answer cut off at `ops_agent.max_tokens` is retried once at `retry_max_tokens`; a
  second cut-off ends the turn with "the answer was cut off", never with the fragment.

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
| R-18 (first), ra-0008 | Three frozen artifacts per run, `make runs ra`, earlier runs offered as context | Superseded by R-18; `knowledge/artifacts` deleted |
| Fixed report headings per kind | `sections: [Summary, Concepts, Examples]` | Superseded by R-19 |
| PRD G1 | Every action passes an always-invoked Gatekeeper | Retired by P-10 |
| `control_loop` package (parked) | FSM + `LoopController` | Built as `loop_control` (`LoopGovernor`) |
| `episodes/`, `make episode` | Per-episode scripts | Retired by P-06 |
