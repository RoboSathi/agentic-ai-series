# Ubiquitous language

The single glossary for the repository: platform, analytics, and each agent. Code identifiers,
log lines, card labels and docs use these words as written; a class named something a viewer
would not recognise from this list is a naming bug. New terms are added here before a module
is designed around them.

**Sections:** [Core](#core) · [Authority](#authority-and-governance) ·
[Memory](#memory-and-context) · [Grounding](#grounding-and-output) ·
[Loop and resilience](#loop-durability-and-resilience) · [Analytics](#analytics) ·
[Evals](#evals) · [Customer Support](#customer-support-agent) · [Research](#research-agent) ·
[Ops](#ops-agent) · [Retired words](#retired-words)

## Core

| Term | Meaning | In code |
|---|---|---|
| **Agent** | Model + Harness. Never "the model" alone. | — |
| **Harness** | Everything around the model that turns a proposal into a governed action. Its removable controls are the **declared features**; the rest is always on. | — |
| **Feature** | A control you can remove and still have a running (worse, or more dangerous) agent. Nine platform features; an agent may declare its own. | `config/features.py` |
| **FeatureSet** | The immutable feature tree, read once in the composition root. | `FeatureSet` |
| **Composition root** | The one function that reads configuration and features and builds the agent; nothing else reads the feature tree. | `graph/builder.py::build_agent` |
| **Withheld** | A feature declared off: its stage is never built and never appears in the graph. | — |
| **Overlay** / **layer** | A YAML file merged over the platform defaults (an agent's config, a profile, `local.yaml`). | `AGENTIC_OVERLAYS` |
| **Profile** (config) | A named posture of an agent's config: `PROFILE=<name>` → `config/profiles/<name>.yaml`. | — |
| **AgentHost** | How an agent reaches the console — title, commands, stage labels, how to build it. Named in config, resolved, never imported. | `interface/host` |
| **Signal** | An opaque string an agent derives from a turn (`intent:REQUEST_REFUND`, `archetype:explainer`); the platform matches it, never interprets it. | skills, dreams |
| **Card** | One stage of a turn as the console draws it; a field appears only if it explains what happened and is read by something. | `StatusEvent` |

## Authority and governance

| Term | Meaning | In code |
|---|---|---|
| **Authority** vs **agency** | Agency is delegated reasoning; authority is the delegated right to act. Claiming authority does not create it. | — |
| **Gatekeeper** | Decides whether the agent has the authority to run the exact call proposed. Not record correctness, not input safety. | `gatekeeper` |
| **Standing** | What a decision means for the principal: **VOID** (denied, nothing queued), **RATIFIABLE** (escalated to a person; approved → **RATIFIED**), **VOIDABLE** (allowed, undoable until it settles), **VALID** (allowed), **UNGOVERNED** (Gatekeeper withheld; still executed and audited). | `Standing` |
| **Authority hierarchy** | provider > org > product > workspace > user task > runtime reminders > tool observations > untrusted content. Only the top ranks may grant authority. | `AuthorityHierarchyResolver` |
| **Mandate** | A deterministic record that this subject may take this action on this target, issued from what the person established — never from model output. | `gatekeeper/mandate.py` |
| **Offered** | The tools a turn exposes to the model. Exposure is not authorization. | — |
| **Surface class** | Locked / editable / append-only / human-controlled. | `tools` |
| **Tier-1 / Tier-2** | Tier-1: deterministic, authoritative, may block. Tier-2: semantic, advisory only, never a veto. | `Tier2Advisor` |
| **Structuring** | Individually permitted actions summing to an unauthorized outcome; defeated by the cumulative ledger. | `CumulativeLedger` |
| **Executor** | Runs an approved call and nothing else: deadline, idempotency, audit. Always on. | `runtime/executor` |
| **Tool path** | The single route every proposed call takes: registry → loop governor → Gatekeeper → Executor. | `tools/path.py` |
| **Effect verification** | Reading the changed thing back; "the executor returned success" and "the money moved" are different claims. | `runtime/verification` |
| **Audit entry** | One decision: trace id, actor, action, authority context, version stamps, decision, standing, rationale. Append-only. | `AuditEntry` |
| **Review item** | Anything the harness hands a person: an escalated refund, a cut reply, a quarantined message, a held fact or skill. Decided with a **comment**, always. | `review` |
| **Kind** (review) | What approving or denying an item does: `memory_fact`, `skill`, `quarantine`, and an agent's own. | — |
| **Subject** | Whose item it is (a customer's id); empty for the agent's own. | `subject_id` |
| **Quarantine** | A message blocked before the model, held as a review item. | `DeadLetterQueue` |

## Memory and context

| Term | Meaning | In code |
|---|---|---|
| **Working memory** | This conversation's turns and current-turn scratch. | `WorkingMemoryStore` |
| **Episodic memory** | What happened — session and run summaries. | `EpisodicStore` |
| **Semantic memory** | What is true — facts, and the knowledge index. | `SemanticStore` |
| **Procedural memory** | How to do things — skills. | `SkillStore` |
| **Agent profile** | An agent's `AGENT.md` (Mission, Scope, Rules): always in the system prompt; never the authority for limits. | `AgentProfile` |
| **Skill** | A procedure as a `SKILL.md`: **authored** (a file in the repo) or **learned** (proposed by a dream, approved by a curator). | `memory/procedural/skills` |
| **Activation** | Which skills a turn carried, as `name@version`, each with why (`always`, `trigger:<signal>`, `lexical:<score>`). | — |
| **Prefetch** | A skill's known first reads, run by the harness before the first model call. | — |
| **Dream event** | What a completed turn or run leaves in the agent's inbox for later consolidation. | `DreamEvent` |
| **Dream** | Consolidation of pending dream events into the three long-term stores, by the agent's `MemoryProfile`. | `Dream` |
| **Curator** | The review that a contested or sensitive memory write waits on. | — |
| **Provenance** | Where a memory came from and how much to trust it. | — |
| **Context assembly** | gather → score → fit → format → order. | `FiveStagePipeline` |
| **Context strategy** | consolidate, isolate, cache, retrieve, offload, reduce. | `context/strategies.py` |
| **Context failure** | rot, distraction, confusion, clash, poisoning. | `context/failure/` |
| **Trust** (chunk) | A chunk's provenance class: `system`, `verified`, `retrieved`, `user`, `model`. | `context/trust.py` |

## Grounding and output

| Term | Meaning | In code |
|---|---|---|
| **Hallucination check** | The claim-level check of a reply, in two **levels**: level 1 token confidence (every agent), level 2 the agent's own grounding. | `hallucination_check` |
| **Atomic claim** | One checkable assertion split out of a reply. | `AtomicClaim` |
| **Claim class** | `record` (a fact about this case), `rule` (how things work; every promise), `conversational` (not scored). | `ClaimClass` |
| **Evidence tier** | One stage of grounding: which evidence kinds it reads, which classes it may ground, whether it may refute. | `hallucination_check.tiers` |
| **Direct check** | The model-free check of ids, amounts and refused calls. | `direct.py` |
| **Fact** / **fact ledger** | A typed value an agent puts in its evidence; the ledger reads claims against facts before any model. | `ledger.py` |
| **State model** | A deterministic state machine claims are judged against (the refund ladder). | `states.py` |
| **Consistent world** | Any current entailment outvotes any contradiction; overtaken reads are dropped first. | `resolver.py` |
| **NOT EXECUTED** | The marker a refused or unverified call carries into grounding. | — |
| **Supported / unsupported / undetermined** | A claim's verdict. Undetermined counts, never blocks — unless the agent says unsupported is a hallucination. | — |
| **Corrected sentence** | A contradicted sentence replaced by the fact's own sentence. | — |
| **Second opinion** | A confirm-only decision model for a claim nothing decided. | `confirmer.py` |
| **Guardrail** | Making output conform before use (schema validation, repair, constrained generation). | `guardrails` |
| **Fail closed** | A reply that cannot be made safe is withheld and handed to a person. | — |
| **NOLLM scoring cascade** | exact assertion → deterministic rule → embedding similarity → NLI → calibrated judge → human. | `scoring` |

## Loop, durability and resilience

| Term | Meaning | In code |
|---|---|---|
| **Tool-use loop** | `propose ⇄ execute`: the model proposes a typed call, the harness decides and runs it. Deliberately not ReAct. | CSA graph |
| **ReAct loop** | A bounded reason–act–observe loop (research direct answer, research worker, ops). | — |
| **LoopGovernor** | The one authority over every agent's loop. | `loop_control` |
| **LoopVerdict** | `continue | suppress | finalize | abort | pause`, with the deciding layer and reason. | `LoopVerdict` |
| **Terminal outcome** | How a turn ended: `answered | finalized_on_cap | aborted | cancelled | escalated | failed`. | `TerminalOutcome` |
| **Step** / **step cap** | One logical model call in a turn; retries and backups are attempts of the same step. The cap is the loop's budget. | — |
| **Durable run** / **thread** | A checkpointed LangGraph run that resumes without repeating committed side effects. | `DurableGraphRunner` |
| **Interrupt** | A deliberate pause waiting on a decision (a reviewer, the user). | `durability/hitl.py` |
| **Stop / Resume / Abort** | The platform's run controls for every durable agent. | `durability/controls.py` |
| **Circuit breaker / bulkhead / saga / idempotency / retry / timeout / pub-sub / hedge** | The resilience patterns (Nygard, *Release It!*). | `resilience` |
| **Compensation** | The undo of one saga step. | `CompensatingSaga` |
| **Admission** | The moment a call gets its rate-limit slot; its deadline counts from here. | `resilience/admission.py` |
| **Rate-limit bucket** / **quota** | A shared request budget for a set of hosts, across processes; a monthly allowance per provider. | `resilience/rate_limiter` |
| **Model chain** | A model, its resilience stack and its hedged backups. | `ModelChainBuilder` |
| **Judge** | A model asked to decide about someone else's output; **abstains** rather than guessing. | `judge` |

## Analytics

| Term | Meaning |
|---|---|
| **Session** | One conversation with one person on one channel. |
| **Task** | One thing the person wants done; may span turns. Opened and closed by the agent through the task API, or implicitly per turn. Left open at session end → **abandoned**. |
| **Turn** | One message in, one reply out (or a resume). Bracketed by the platform. `turn_id` is the turn's trace id. |
| **Turn status** | `completed`, `crashed`, `cancelled` (platform-defined; not an outcome). |
| **Outcome** | How a task ended, in the agent's own words, flagged `success` and `terminal`. Reserved: `unclassified`, `crashed`, `cancelled`, `abandoned`. |
| **Resolution** | A terminal task with a successful outcome. |
| **Model call** (attempt) | One request to one model; a **backup** attempt is one served after the role's primary. |
| **Tool proposal** and **stage** | A call the model asked for, and how far it got: `invalid`, `unknown`, `not_offered`, `refused`, `duplicate`, `executed`. **Tool-call validity** = share that reached `executed`. |
| **Check** | One invocation of a guard or classifier (`pii`, `injection`, `jailbreak`, `moderation`, `content_safety`, `grounding`, `nli`, `embedding`, `rerank`, `intent_model`, `logprob`), with a backend, a model and a verdict: `pass`, `flag`, `block`, `skipped`, `error`, `fail_open`. |
| **Screen coverage** | Whether a declared screen actually ran on a turn. |
| **Groundedness** | Supported ÷ checked claims. |
| **Decision** | One Gatekeeper ruling with its standing; in the audit trail, not a Tier-1 metric. |
| **Tier 1 / 2 / 3** | Fleet metrics from platform choke points / an agent's own tables / eval results. |
| **Fact** | One row at one grain, appended once by its choke point, never updated. |
| **Source** | One analytics dataset: a DuckDB and the Parquet export of its curated views (`live`, `dummy`). Readers read only the Parquet. |
| **Semantic layer** | The YAML that defines each curated view's SQL, its column meanings and the metric catalogue. |

## Evals

| Term | Meaning |
|---|---|
| **Suite** | `golden` (graded by the oracle and DeepEval) or `security` (Promptfoo). |
| **Section** | One grader over a suite, with its own gate and report: `oracle`, `deepeval`, `promptfoo` (and `intent`). |
| **Oracle** | The code grader: reads the outcome in the database against ground truth; zero model calls. |
| **Golden case** | Turns plus an agent-owned payload, hashed into `case_hash`. |
| **Transcript** | Replies, tool calls with their rulings and standings, evidence, side effects, usage. |
| **Core / full** | The default subset of cases, attacks and metrics, or all of them. |
| **Hard / soft gate** | 100% required / a pass rate. |
| **Trial** | One of k runs of a case; the report shows **pass@k** (some trial clean) and **pass^k** (every trial clean). |
| **Agent fingerprint** / **grader fingerprint** | Content hashes that decide when a transcript or a score can be reused. |
| **Intent bake-off** | Intent classifiers scored side by side, offline. |

## Customer Support agent

| Term | Meaning |
|---|---|
| **User code** | The seven-character credential a customer types; deliberately not the enumerable user id. |
| **Order under discussion** | The order the conversation established; writes are scoped to it. |
| **Order menu** | The numbered list a customer picks an order from. |
| **Intent** | `ORDER_STATUS`, `ORDER_DETAILS`, `LIST_ORDERS`, `REQUEST_REFUND`, `REFUND_STATUS`, `CANCEL_ORDER`, and `UNKNOWN` — each with a mode (`read` / `write`) and its tools. |
| **Disposition** | supported, needs clarification, or out of scope. |
| **Deterministic read** | A read intent answered straight from the database, with no model call. |
| **Unplaced turn** | A turn no intent claimed (`route:llm_fallback`); gets no skill by default and reads-only tools. |
| **Follow-up** | A message that restates nothing and continues the last subject ("what about 10010?"). |
| **Refundable** | Delivered, or cancelled before delivery. |
| **Direct limit** | The most the agent may refund without a person; a database row, never config. |
| **Issued / settled** | Paid and still voidable / completed and final. |
| **Void** | Reversing an issued refund before it settles. |
| **Refunds ledger** | What was actually paid, separate from an order's `refund_status`. |
| **Output quality gate** | Plan lookup → one deterministic critique → at most one repair → independent score. A single pass, not a loop. |

## Research agent

| Term | Meaning | In code |
|---|---|---|
| **Direct answer** | A plain question answered in one turn by a ReAct loop over read-only tools, checked against what they returned. | `graph/answer_*.py` |
| **Research kind** (archetype) | `decision_support`, `landscape_brief`, `fact_verification`, `explainer`, `general`; picks checklist, rubric, stop rule and report. | `domain/scope.py::Archetype` |
| **Latest by default** | A question that names no date asks for what is current. | `research/recency.py` |
| **Time window** | "in the last month" → the Brief's `recency_days`, enforced by the worker. | `scoping/window.py` |
| **Coverage checklist** | Per-kind interview items: `open`, `covered`, `waived` or `delegated`. | `scoping/checklists.yaml` |
| **Known unknown** | A scope item waived or left open at `/done`; the rubric and report must acknowledge it. | `KnownUnknown` |
| **Scope Brief** | The typed, frozen output of scoping. | `ScopeBrief` |
| **Rubric** / **rubric item** | The acceptance criteria; each item has a check (`deterministic`, `nli`, `judge`), threshold, `must_pass`, origin and mapping. | `domain/rubric.py` |
| **Research contract** | Brief + Rubric + Budget, frozen at `/approve` and identified by its canonical hash. | `Contract` |
| **Budget** | The run's hard caps: searches, fetches, tokens, seconds, rounds, DAG nodes, gap-fill rounds, concurrency. | `domain/budget.py` |
| **Plan DAG** / **sub-question** | The planner's decomposition; every node traces to a rubric item. | `domain/plan.py` |
| **Round** / **batch** | One plan → research → reflect cycle / one set of concurrently runnable nodes (a superstep). | — |
| **Evidence ledger** | Run-scoped, append-only: claim → URL → verbatim span → page hash → retrieved at. | `hallucination_check.evidence.EvidenceLedger` |
| **Verbatim-span rule** | A span is admitted only if it is a substring of the page this run fetched. | `VerbatimSpanValidator` |
| **Provider summary** | Text a search provider writes about its results; untrusted, never evidence. | `SearchResponse.provider_summary` |
| **Gap** / **gap-fill** | A rubric item not yet satisfied / one bounded round of targeted research for must-pass gaps. | — |
| **Stop rule** | A kind's "done" test (every option scored; no new entities; every claim decided; every concept covered). | `research/stop_rules/` |
| **Independent audit** | The fresh-context verification that sees only `(contract, report, ledger_snapshot)`. | `IndependentAuditor` |
| **Writing fault** / **revision** | An audit failure of how the report is written, fixed by revising against the findings; bounded. | `run_nodes.revise` |
| **Cross-check** | A subject supported by entailed claims from at least two independent domains, none contradicted. | `audit/checks.py` |
| **Run event** | What a run's critics found, left in the dream inbox scoped to the run's kind. | `RunEventWriter` |
| **Coverage & Limits** | The report section, built in code, listing every unmet must-pass item with what was tried and why the run stopped. | `synthesis/limits.py` |
| **COMPLETE / PARTIAL** | Every must-pass item met and no budget exhausted / anything less. | — |
| **Hostile page** | A page carrying instructions aimed at the reader (override, exfiltration, fake authority). | — |

## Ops agent

| Term | Meaning |
|---|---|
| **Figure check** | Level-2 grounding for ops: every number in the answer must appear in a query result. |
| **Envelope** | The surface, side effects and permission this deployment grants an MCP server's tools, whatever the server says. |
| **Query guard** | The SQL wall: one SELECT over curated views, no table functions, an outer LIMIT. |

## Retired words

| Word | Use instead | Why |
|---|---|---|
| mart | source | ADR P-24 |
| procedure (user-visible), charter | skill, agent profile | ADR P-12 |
| goldfish audit | independent audit | renamed 2026-10-02 |
| `basic_sanity` | `deterministic_checks` | renamed 2026-10-10 |
| answered / escalated / unanswered as cross-agent outcomes | the agent's own outcomes | they were support's words |
| Q&A mode | direct answer | ADR R-01 |
| control_loop, LoopController | loop_control, LoopGovernor | built under the new name |
| fleet page | Agent Platform dashboard | ADR P-26 |
