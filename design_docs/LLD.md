# LLD — Agentic Harness Platform

Component-level design: for each package, what it owns, its public interface, its
configuration, and its seams. Part A is the platform (`src/agent_platform/`), in the order a
reader should meet it; Part B is each agent; Part C the invariants no single module owns.
Architecture is in [HLD.md](HLD.md); the reasons in [ADR.md](ADR.md).

Every package's own `README.md` is a short pointer to its section here.

**Contents**

- Part A — Platform
  [config](#a1-config) · [logging](#a2-logging) · [core](#a3-core-primitives-persistence) ·
  [llm](#a4-llm) · [judge](#a5-judge) · [scoring](#a6-scoring) · [tools](#a7-tools) ·
  [loop_control](#a8-loop_control) · [runtime](#a9-runtime-executor-verification-scheduling) ·
  [gatekeeper](#a10-gatekeeper) · [durability](#a11-durability) · [security](#a12-security) ·
  [deterministic_checks](#a13-deterministic_checks) · [intent](#a14-intent) ·
  [context](#a15-context) · [memory](#a16-memory) · [hallucination_check](#a17-hallucination_check) ·
  [guardrails](#a18-guardrails) · [resilience](#a19-resilience) ·
  [observability](#a20-observability) · [audit](#a21-audit) · [review](#a22-review) ·
  [knowledge](#a23-knowledge-web-mcp) · [interface](#a24-interface-tui-commands-presentation-host) ·
  [evals](#a25-evals) · [analytics](#a26-analytics)
- Part B — Agents
  [Customer Support](#b1-customer-support-agent) · [Research](#b2-research-agent) ·
  [Ops Analyst](#b3-ops-agent)
- [Part C — Cross-subsystem invariants](#part-c--cross-subsystem-invariants)

---

# Part A — Platform

## A1. config

**Owns.** Every value in the repo. Loads the layered YAML, validates it into typed settings,
and owns the feature tree that decides which harness controls an agent is given.

- **Layers** (deep-merged, later wins): platform `config/config.yaml` → agent overlays
  (`AGENTIC_OVERLAYS`, `:`-separated; a profile is one more layer) → run overlays
  (`use_overlay`) → `config/local.yaml` (gitignored) → `SECTION__FIELD` environment variables.
  Mappings merge; scalars and lists replace (a list that merged element-wise could never be
  shortened). YAML anchors and merge keys (`<<:`) are supported; `x-`-prefixed top-level keys
  hold anchors only.
- **Section** — one top-level key validated by one `YamlBackedSettings` class, env prefix
  `SECTION__`. Bare provider keys (`nvidia_api_key`, `tavily_api_key`, `langchain_api_key`) are
  plain env vars and never appear in a settings dump.
- **Interface.** `get_settings()` / `reload_settings()` (the `AppConfig` singleton);
  `YamlBackedSettings`; `use_overlay` / `clear_overlays` / `config_sources`; `FeatureSet`,
  `active_features`, `set_active_features` (`features.py`); `resolve_path` / `REPO_ROOT`
  (`paths.py` — paths resolve against the repo, never the CWD); `load_environment`
  (`environment.py` — `.env` fills the process environment once; an exported variable wins);
  `VersionManifest` (`versions.py` — independently versioned components).
- **Feature tree.** The nine platform features (ADR P-07); an agent may declare its own
  (the research agent's `rubric`, `reflection`, `independent_audit`, …). Read once in the
  composition root. `/harness` moves the live `FeatureSet`; `/config` is read-only.
- **Guard.** `tests/customer_support_agent/test_config_surface.py`: no YAML key that no
  settings model declares.

## A2. logging

**Owns.** Structured per-subsystem logs to files (always) and the console (optional), and the
`@instrument` decorator that records what a traced call was given and returned — in full,
never truncated by default.

- **Interface.** `get_logger(subsystem)`, `log_directory()` (`core.py`); `@instrument`,
  `@logged` (`decorators.py`); `get_current_trace_id` / `set_current_trace_id`
  (`trace_context.py`).
- **Layout.** `logs/<agent id>/<subsystem>.log`; `run_label` adds a sub-folder for runs that
  are not the live agent. `payload_max_chars: 0` keeps everything.
- **Config.** `logging:` — levels, console/file switches, `dir`, `file_format`,
  `payload_max_chars`, rotation.

## A3. core (primitives, persistence)

**`core/primitives`** — layer-0 value types, no dependency on any other platform package:
`Money` (integer cents; never a float), `canonical_hash` / `text_hash`, `aes_key`, PII token
shapes, an injectable clock.

**`core/persistence`** — thread-safe SQLite access and the platform's own per-agent tables
(agent identity, permissions, budgets, review queue, PII vault). An agent's business tables
and repositories live in the agent.

- **Interface.** `SqliteConnectionPool` (`execute`, `execute_many`, `execute_script`,
  `query_one`, `iter_rows`); `PlatformStore` (Protocol), `SqlitePlatformStore`; `Repository`
  (Protocol); `IN_MEMORY`.
- **Config.** `persistence.db_path` — a neutral default; each agent overlays
  `db/<package>/agent.db`.
- **Seam.** A pool is injectable everywhere (`build_agent(pool=...)`); tests use `:memory:`.

## A4. llm

**Owns.** The only place a model provider is called.

- **Clients.** `LLMClient` (ABC: `complete`, `health_check`, `model`, `tier`);
  `OpenAICompatibleClient` / hosted and grammar-constrained adapters (`llm.providers.<key>.adapter`);
  `CascadingLLMClient`; `ResilientLLMClient` (one `ResilientCaller` per model).
- **`ModelChainBuilder`** (`chain.py`) — the only place a chain is composed: per-model
  resilience, hedged backups (`hedge_after_seconds`), `RecordedClient` (writes a
  `fact_model_call` per attempt), `SteppedClient` (counts steps), the PII boundary when `pii`
  is declared. `llm.chain` configures the default chain; `llm/roles.py` (`RoleModelRouter`)
  resolves an agent's named roles and refuses two roles that must differ resolving to one model.
- **Embeddings and encoders.** `HostedEmbeddingClient`, `HashingEmbeddingClient` (offline);
  local NLI (`local_nli.py`), local reranker (`local_reranker.py`), local summarizer, local
  Laya, local e5 embeddings — all through the shared `local_models.py` loader, which downloads
  into `local_models/` on first use.
- **HTTP.** `create_http_client` / `create_sdk_http_client` (`http.py`): `httpx` clients whose
  transport is the cross-process rate limiter.
- **Metering.** `llm/metering.py` — the context-variable usage sink (ADR P-23).
- **Prompts.** `llm/prompt.py` (`PromptBuilder`) builds a request from named sections; drawn as
  the CONTEXT card.
- **Probes.** `local_probes.py` (`make models`), `cli.py` (`make model-check`, `BENCH=1`).
- **Config.** `llm:` — `provider`, `providers.<name>` (base URL, model, timeout, cost rates,
  key env vars), `chain`, `logprobs`, `tokenizer`, `context_windows`, `embedding`, `encoders`,
  `intent_classifier`, `check`. Agents add `llm.roles`.

## A5. judge

**Owns.** The LLM judge every eval grader and the research audit build on.

- `create_judge_client()` — the `judge:` model and its backups, through `ModelChainBuilder`.
- `LlmJudge[Request, Answer, Verdict]` — Template Method: `judge(request)` calls the `_answer`
  hook, then `verdict`; any failure goes to `abstain` (never a silent pass or fail);
  `models_used()` says who judged.
- `ModelJudge` — implement `render(request) -> messages`; the platform judge model, the grading
  rules (`prompts/judge_rules.md`: evidence only, reason before verdict, quote what earns a pass,
  say so when undecidable) and bounded schema repair are supplied.
- `ask_validated` / `ask_text` — one judge call for a framework with its own loop (DeepEval).
- `JudgeCalibrator` — Cohen's κ against a second rater.
- **Config.** `judge:` — `provider`, `model`, `fallbacks`, `timeout_seconds`,
  `hedge_after_seconds`, `temperature`, `max_tokens`, `max_evidence_chars`, `validation_mode`,
  `kappa_acceptable`.

## A6. scoring

**Owns.** The NOLLM scoring cascade — runtime infrastructure, not an eval. Retrieval relevance,
tool fitness, grounding and memory consolidation resolve claims here, cheapest tier first:
exact assertion → deterministic rule → embedding similarity → NLI entailment → calibrated LLM
judge → human adjudication. A tier that cannot decide declines so the next may try.

- `create_scoring_cascade()`, `NollmScoringCascadeImpl` (`nollm_scorer_ratio`, `tier_counts`);
  one scorer class per tier; `EntailmentModel`, `LexicalEntailmentModel` (offline),
  `create_entailment_model`.
- A leaf package: imports configuration, embeddings and logging, nothing that imports it.
- Tier 5 declines until a `CalibratedJudge` is wired in.

## A7. tools

**Owns.** Typed tool definitions in a namespaced registry, retrieval-based selection, and the
single tool path.

- `ToolRegistry` / `NamespacedToolRegistry`; `ToolDefinition`, `ToolCall`, `ToolResult`,
  `ToolHandler`, `ToolSource` (`types.py`).
- `ToolRetrievalSelector` — offers a turn only relevant tools, none below a relevance floor.
- **Model names** (`names.py`): the model and the cards see `process_refund`, never
  `support__process_refund`; qualified names stay in policy, audit, facts and evals.
- **`ToolPath`** (`path.py`) — every model-proposed call: registry (`unknown`, `invalid`) →
  loop governor (`duplicate`) → Gatekeeper (`not_offered`, `refused`) or UNGOVERNED → Executor
  (`executed`: succeeded or failed). Records the stage reached as `fact_tool_call` (ADR P-11).
- **Surface classes** — locked, editable, append-only, human-controlled.
- **Config.** `tools:` — `enabled`, `selection_k`, `selection_min_score`, `model_names`.

## A8. loop_control

**Owns.** One authority over every agent's loop: `LoopGovernor`.

- Governs the step cap, budgets (tool calls, tokens, cost, wall clock), a deadline that
  cancels a call still running, repeat detection (exact → canonical → semantic), cycle
  detection, convergence (new evidence per step) and a historical baseline.
- Answers with one contract, `LoopVerdict`: `continue | suppress | finalize | abort | pause`,
  the layer that decided, its reason, every budget line and every check. Routers read only
  its decision (`route_on_verdict`); every agent draws the same LOOP CONTROL card from it.
- **Lifecycle.** `begin_turn` → per step `begin_step` → `call_limits` / `deadline` →
  `after_model` → per action `before_action` → `after_action` → `finish(outcome)`;
  `restore(state)` continues a checkpointed run.
- `TerminalOutcome`: `answered | finalized_on_cap | aborted | cancelled | escalated | failed`.
- `LoopState` (the governor's whole memory, kept in the agent's checkpointed state),
  `LoopStateMachine` (legal phase transitions).
- `BudgetGuard` (`budget.py`) — synchronous reserve → commit/release on named counters plus a
  wall clock, for parallel workers.
- `build_loop_governor(budget, facts=, agent_id=)` — composition root only. Caps are each
  agent's own config, mapped onto `LoopBudget`.

## A9. runtime (executor, verification, scheduling)

**`runtime/executor`** — the deterministic execution boundary. Runs a tool call the
Gatekeeper approved, and nothing else: deadline, idempotency, lifecycle events and an audit
hook around the handler. Only an `ApprovedToolCall` reaches it. Always on — with the
Gatekeeper withheld the call carries `ungoverned` (standing UNGOVERNED).
`ToolExecutor` (Protocol), `ToolExecutorImpl`, `current_execution_key()` (the key a call runs
under, for a handler whose effect can later be voided), `RegistryToolHandler`
(self-implementing tools such as MCP's).

**`runtime/verification`** — did the side effect happen? `ReadBackVerifier` reads back what
should have changed through an agent's `EffectSource` and compares it with the
`IntendedEffect`; a result that cannot be verified never becomes authoritative evidence.
`acceptance.py` — `AcceptanceCheckRegistry` of named checks by kind (deterministic, NLI,
judge), used by the research audit.

**`runtime/scheduling`** — dependency-aware concurrent scheduling. `TaskGraph` (validation,
cycle finding, waves) and `AsyncDagScheduler` (bounded concurrency; a node's failure blocks its
dependents and nothing else).

## A10. gatekeeper

**Owns.** Whether the agent has the **authority** to run the exact call the model proposed.
Not whether the action is correct for the record (the tool checks that, once, with a
conditional write) and not whether the input is safe (screening does that). Calls no model,
touches no database per call, audits every decision (ADR P-09).

**Questions, in order; the first failure decides.** Is the agent enabled → was the tool offered
this turn → does the call target the record the conversation established (**mandate**) → does
the role hold the tool's permission (**RBAC**) → does the instruction's source rank allow it
(**authority hierarchy**) → is it within the agent's own bounds (its business rules, e.g. a
refund limit) → where a deployment has one, the cumulative budget.

| Standing | When | Effect |
|---|---|---|
| VOID | denied | nothing runs, nothing is queued |
| RATIFIABLE | escalated (beyond the agent's bounds) | queued for a person; approval is RATIFIED |
| VOIDABLE | allowed, and the tool declares `voidable` | runs; undoable until it settles |
| VALID | allowed | runs |
| UNGOVERNED | the Gatekeeper is withheld | runs through the Executor; audited as such |

- **Interface.** `Gatekeeper` (Protocol), `GatekeeperImpl`; `AuthorityHierarchyResolver`,
  `RoleAuthorizer` (`hierarchy.py`); `CumulativeLedger` (`ledger.py`, the structuring defence);
  `DeterministicBusinessRules` (Protocol, `AllowAllBusinessRules`); mandates (`mandate.py`);
  `Tier2Advisor` (a semantic advisor that can flag, never allow); `CheckReport`,
  `AuthorityContext`, `AuthorityDecision`, `Verdict`, `Standing`.
- **Config.** `gatekeeper:` — `authority_grant_max_rank`, `authority_hierarchy`, `ledger`,
  `mandate`, `tier2`, `surface_policy`. Built only when `features.gatekeeper`.
- **Seam.** An agent supplies its business rules and mandate policy.

## A11. durability

**Owns.** Runs that survive failure. Every turn is a checkpointed LangGraph thread; a failed
node resumes without repeating committed side effects; a human-in-the-loop pause is the same
mechanism.

- `create_checkpointer(state_types=, path=)` → `SqliteCheckpointer` (AES-encrypted with
  `LANGGRAPH_AES_KEY`; sync SQLite bridged for async).
- `DurableGraphRunner` — `start`, `resume(thread, decision)`, `abort`, `values`, `history`.
- `RunRegistry` — run records, `resumable`, `recover_orphans`, `purge_expired`.
- `hitl.await_decision` — the interrupt a graph pauses on; `IsolatedGraphRunner`
  (`isolated.py`) — a sub-graph that accepts only declared input keys
  (`UndeclaredInputError`).
- **Run controls** (`controls.py`, `stop_report.py`): Stop prints the agent's state (each
  agent supplies `describe_progress(values)`), then Resume or Abort; Abort keeps the run's
  record. Every `DurableAgent` gets them.
- **Config.** `durability:` — `db_path` (agent overlays `db/<package>/checkpoints.db`), `mode`,
  `retention_days`, `preview_chars`, `encryption`.

## A12. security

**Owns.** Who is acting with what permissions, PII, prompt injection and content moderation —
each deterministic first, paired with an optional hosted guard and its fallbacks.

- `SecuritySuite` (`suite.py`) — one facade over every control; `MessageScreener`
  (`screening.py`) — the shared front-door screen.
- **Auth** (`auth/`): `Authorizer`, `AgentPolicy` (role, authority rank, permissions,
  per-action and window limits — read from the agent's database), `AgentPolicyRepository`,
  `Principal`, `Permission`.
- **PII** (`pii/`): `PIIFilterImpl` (detect, redact, minimise) over Presidio + local spaCy,
  regex recognizers as fallback; `PiiBoundary` (tokenises provider traffic); the vault
  (`HMAC` tokens, AES-GCM values, retention) and `reveal` (`make pii-reveal`, audited first).
- **Prompt injection** (`prompt_injection/`): phrase rules and instruction/data separation
  (untrusted text wrapped in delimiters), `HostedJailbreakGuard`, `ChatJailbreakGuard` (a chat
  model prompted as the classifier).
- **Content moderation** (`content_moderation/`): term rules, `HostedContentSafetyGuard`.
- `GuardChain` — a guard and its fallbacks asked in turn; the first verdict that is not
  `unavailable` wins. `security.guards.fail_open_on_unavailable: false` blocks the turn only
  when every one is unavailable.
- **Config.** `security:` — `auth`, `pii` (engine, entities, vault), `prompt_injection`,
  `content_moderation`, `guards`. Stages exist per `features.pii`, `prompt_injection`,
  `content_moderation`.

## A13. deterministic_checks

**Owns.** The contract for checks that run before any model is asked: does the thing named
exist, is it the caller's, is the request possible? A request in, findings out; every
concrete checker belongs to an agent (ADR P-08).

- `DeterministicChecker` (Protocol): `check(request) -> DeterministicCheckResult`;
  `DeterministicCheckRequest` (`subject_id`: opaque), `DeterministicCheckFinding`
  (applicable, passed, reason, evidence), `DeterministicCheckResult`.
- No config section; the stage exists when `features.deterministic_checks` is on.

## A14. intent

**Owns.** Running a classifier and reporting candidates with scores. Which intents exist,
their tools and thresholds are an agent's business.

- `IntentClassifierInterface` (Protocol); `HuggingFaceIntentClassifier` (a fine-tuned
  sequence classifier through a label mapping); `LayaIntentClassifier` (a decision model asked
  one choice question over the agent's label descriptions, injected at build time);
  `create_intent_classifier(choices, instructions)`.
- `IntentAssessment` (chosen intent, disposition `supported | clarify | out_of_scope`,
  candidates, source), `IntentCandidate`.
- **Config.** `llm.intent_classifier:` — `enabled`, `provider` (`huggingface_local` |
  `laya_local`), model, `local_model_dir`, `download_if_missing`, `max_length`, `laya:`.
  Load before an event loop opens (macOS tokenizer start-up).

## A15. context

**Owns.** What the model is handed: which candidates enter, how they rank, what is reduced,
offloaded or isolated to fit the token budget, and whether the result shows a known failure.

- **Pipeline.** `FiveStagePipeline`: gather → score → fit (strategies, then cut) → format →
  order. `ContextAssembler`; `CandidateSource` (Protocol) — e.g.
  `WorkingMemoryContextSource`, the knowledge-base source.
- **Strategies** (`strategies.py`, `StrategySelector`): consolidate, isolate, cache (always),
  then retrieve, offload, reduce (while over budget). Offloaded text waits in session working
  memory and comes back through the `fetch_offloaded_context` tool.
- **Retrieval.** `LexicalRetriever` (BM25), `DenseRetriever`, `HybridRetriever`,
  `GraphRetriever`; `CrossEncoderReranker` (local).
- **Trust.** A chunk's provenance class (`system`, `verified`, `retrieved`, `user`, `model`)
  decides who wins a clash.
- **Failure detection** (`failure/`, one module per mode): poisoning (injection, or an
  unverified claim a verified source contradicts — prose facts only), clash (local NLI),
  distraction (similarity to the task), confusion (near-copies, too many tools), rot (quality
  falling as the window grows).
- **Config.** `context:` — budget, retrieval, rerank, strategies, similarity, trust,
  `failure_detection`. Exists per `features.context` (+ `rerank`, `failure_detection`).

## A16. memory

**Owns.** The four CoALA stores — working, episodic, semantic, procedural — all real, plus
governed writes, dreams, skills, a deterministic recall tier and document RAG.

- **Files** (`files/`, ADR P-13): `MemoryDirectory`, frontmatter records under `.history/`,
  regenerated views, `MEMORY.md` per store, a derived FTS5 index. Root per agent:
  `data/<package>/memory/`.
- **Stores.** `WorkingMemoryStore` (this conversation), `EpisodicStore` (what happened),
  `SemanticStore` (what is true; also the knowledge index), `SkillStore` (how to do things).
- **Governance.** `MemoryFacade` (the governed entry point); `MemoryWritePipelineImpl`:
  extract → classify → provenance → conflict → confidence → authorize → route (auto-commit,
  curator queue, discard); `MemoryCommand`, `MemoryPrincipal`, `MemoryAuthorizer`. Categories
  that may never be written from conversational assertion are configured.
- **Dreams** (`dream/`): `DreamEvent`, `FileDreamInbox` (`pending/` → `consolidated/`),
  `TurnRecorder`, `Dream`, `MemoryProfile` (three consolidators; `MemoryProfile.default()`),
  `RecurringSignalSkills`. Composition for any agent is in `interface/host/memory.py`.
- **Skills** (`procedural/skills/`, ADR P-12): `SkillDocument`, `SkillCatalog` (sync files →
  store), `SkillSelector` (always ∪ triggered, else BM25, capped), `AgentInstructions`
  (`select`, `announce`, `render`, `compose`), `build_agent_instructions`.
- **RAG** (`semantic/rag/`): chunking, sqlite-vec + FTS index, reciprocal-rank fusion,
  retriever, knowledge-base context source.
- **Config.** `memory:` — `files`, `dream` (retention, `skill_threshold`, refusal markers,
  learned-skill wording), `semantic.rag`, `write_pipeline`, provenance and retrieval weights,
  `recency_half_life_days`, `governance`, `deterministic`. `skills:` — `profile_path`,
  `skills_dir`, `selection`, `render.sections_sent`, `max_prompt_chars`.
- **Commands.** `make dream | show-memory | clean-memory | memory-reindex <agent>`; `/memory`,
  `/dream`.

## A17. hallucination_check

**Owns.** Claim-level grounding of a draft reply against the evidence the agent actually had
(ADR P-16, P-17).

- **Levels** (`levels.py`). Level 1, token confidence (`confidence.py`), for every agent; level
  2, the agent's grounding. Both travel in the one `hallucination_check` event.
- **Claims** (`claims.py`): split into atomic claims (a line break ends a claim; markdown
  stripped; a lead-in line claims nothing), each classed `record | rule | conversational`.
- **Resolution order.**
  `record → direct check → fact ledger → state model → tool_result (NLI) → second opinion*`;
  `rule → direct check* → fact ledger → policy → system prompt → memory (NLI)`;
  `conversational → not checked`. (* only for a date or number no evidence contains; the second
  opinion is confirm-only and opt-in.)
- **Direct check** (`direct.py`): ids and amounts the record lacks, refused calls
  (`NOT EXECUTED`). **Fact ledger** (`ledger.py`): typed facts from `fact_chunk`; comparisons
  computed, flags affirmed or denied; any disagreement refutes (`FACT_MISMATCH`).
  **State model** (`states.py`): the refund state machine, judged against the latest status.
- **Evidence** (`evidence/`): `observations.py` turns tool results into one premise per
  sentence in call order, plus passage windows (`tool_evidence`, `Observation`);
  `ledger.py` / `validator.py` hold the append-only `EvidenceLedger` with the verbatim-span rule
  (`VerbatimSpanValidator`, `NonEmptySpanValidator`).
- **Resolver** (`resolver.py`): drops overtaken reads, then one consistent world — any current
  entailment outvotes any contradiction.
- `DetectionLadder`, `GroundingReasoningDiagnoser`, `TrustCalibrationTracker` (`checker.py`);
  `ClaimConfirmer` (`confirmer.py`).
- **Config.** `hallucination_check:` — thresholds, premise width (`passage_window`,
  `joined_premise`), `tiers`, classifier vocabulary, refund state machine, `facts`,
  `confidence`, `unsupported_is_hallucination`, `undetermined_marker`, `confirmer`.

## A18. guardrails

**Owns.** Model output conforms before it is used: validate against a schema, repair or reject,
or constrain generation. `ResponseGuardrail` runs a customer-facing reply through the
hallucination check with a bounded repair loop that fails closed to a person.

- `PydanticValidator`, `LlmConstrainedGenerator`, `fail_loud` (`guards.py`);
  `ResponseGuardrail`, `ResponseGuardrailResult`, `CustomerResponse` (`response.py`);
  `OutputValidator`, `ConstrainedGenerator` (Protocols), `GuardrailViolationError`.
- **Config.** `guardrails:` — `enabled`, `mode` (`reject | auto_repair | constrained`),
  `max_repair_attempts`, `constrained_backend`. The grounding half exists per
  `features.hallucination_guard`.

## A19. resilience

**Owns.** Nygard's patterns, implemented and composed by `ResilientCaller` in one fixed order:
bulkhead → breaker → retry → timeout → operation.

- `CircuitBreakerImpl`, `ExponentialBackoffRetry` (retrying a side effect without an
  idempotency key is refused), `AsyncTimeoutGuard`, `ResourcePoolBulkhead`,
  `SqliteIdempotencyGuard`, `CompensatingSaga` / `SagaWorkflow` (compensates in reverse, also on
  cancellation), `InMemoryPubSub`, `DeadLetterQueue` (quarantines as review items),
  `HedgedCall` (`hedge/`).
- **Admission** (`admission.py`): a call's deadline pauses while it waits for a rate-limit slot
  (ADR P-21).
- **Rate limiter** (`rate_limiter/`): `SlidingWindowRateLimiter` over `SqliteRateWindowStore`
  (shared by every process); `RateLimitedTransport` (httpx) and `RateLimitedHttpx2Transport`
  (the OpenAI SDK); per-bucket monthly quotas (`QuotaExhaustedError`, never retried).
- **Config.** `resilience:` — one block per pattern with its own `enabled`; `rate_limit`
  (`store_path`, `window_seconds`, `buckets.<name>.{max_requests, hosts, monthly_quota}`).
  `RPM=<n>` overrides the NVIDIA bucket for one invocation.

## A20. observability

**Owns.** Everything the harness reports about itself while it runs.

- **Feed and cards.** `StageReporter`, `StatusEvent`, `InMemoryLiveStatusFeed`; shared card
  field builders (`llm_fields.py`, `tool_fields.py`), `LoopReporter` ("Step N"), the CONTEXT
  card (`context_card.py`), model-chain cards (`model_observer.py`). An empty field is dropped;
  a `quiet` event is logged and counted but draws no card.
- **Facts** (ADR P-24): `FactStore`, the Tier-1 fact models (`facts.py`); the turn bracket and
  task API (`units.py`: `open_task`, `close_task`, `mark_cancelled`, `end_session`,
  `OutcomeCatalog`); the check recorder (`checks.py`: `record_check`, `record_screen`,
  `recorded_dependency`).
- **Store.** `DuckDbObservabilityStore` — `db/agent_platform/observability.duckdb` (one writer).
- **Tracing.** `LocalTracer`, `traced`; `LangSmithExporter`, `turn_trace` (ADR P-27).
- **SRE.** `CostAggregatorImpl`, `SLORegistry` / `SLOTracker`, `LittlesLawQueueModel` (review
  staffing), `PostmortemRegister`.
- **Config.** `observability:` — `enabled`, `langsmith`, `persist_traces`, `slo_targets`,
  `review_queue`, `llm_purposes`.

## A21. audit

**Owns.** The append-only decision trail: who asked, for what, under which authority and
policy version, the verdict and its standing. It decides nothing; it remembers.

- `AuditLedger` (Protocol: `record`, `count`, `by_decision`, `reconstruct(trace_id)`);
  `SqliteAuditLedger`, the DuckDB ledger, `NullAuditLedger`; `create_audit_ledger`;
  `AuditEntry`, `Decision` (`allow | deny | escalate`).
- `AuditTrail` (`view.py`) merges the shared DuckDB ledger and the agent's own SQLite ledger;
  one renderer serves `/audit` and `make audit <agent>` (filters `DECISION`, `STANDING`, `KIND`,
  `SINCE`, `N`; `TRACE=<id>` as a tree). Tokens only.
- **Config.** `audit:` — `enabled`, and the version stamps on every entry
  (`policy_version`, `data_version`, `prompt_version`, `tool_schema_version`).

## A22. review

**Owns.** One human-review queue for every agent (ADR P-14).

- `ReviewQueue` (`platform_review_queue` in the agent's database), `ReviewService`,
  `ReviewHandler`, `ReviewError`; platform handlers `MemoryReviewHandler`, `QuarantineHandler`.
- **Kinds.** Platform: `memory_fact`, `skill`, `quarantine`; an agent registers its own.
- **Doors.** `/review` (`interface/commands/review.py`), `make review <agent>`
  (`interface/commands/review_cli.py`), the Review screen (`ctrl+r`,
  `interface/tui/review_screen.py`).
- **Config.** `review:` — table, `min_comment_chars`, kinds, audit action and trace templates,
  console reviewer, every word shown.

## A23. knowledge (web, mcp)

**`knowledge/web`** — one gateway for agents' web access, live / record / replay.
`create_web_gateway()`; `WebGateway` (observer-notified `search`, `fetch`); `SearchRouter`
over `SearchProvider` adapters (`DuckDuckGoProvider`, `TavilyProvider`); `SearchCache` (on
disk, 24 h); `HttpFetcher` with a reader fallback for JS-only pages; `FixtureStore`;
`SearchResponse.provider_summary` (display only). Live calls go through `ResilientCaller` and
the rate limiter. Config: `web:` (mode, providers, routing, cache dir, fixtures dir).

**`knowledge/mcp`** — Model Context Protocol, by side of the wire (ADR P-25).
`common/` (types, stdio), `client/` (`SdkMcpTransport`, `SdkMcpSession` — an Active Object
owning the SDK client in its own task; `GovernedMcpToolSource`, `ConfiguredGovernancePolicy`,
schema-built argument models, the MCP SERVER card), `server/analytics/` (`build_server`;
`python -m agent_platform.knowledge.mcp.server.analytics`). Config: an agent's
`mcp.servers[]` (name, transport, command — `{python}` is this interpreter — namespace,
envelope) and `mcp.default_envelope`.

## A24. interface (tui, commands, presentation, host)

**`interface/host`** — the shell ↔ agent contract. `AgentHost` (title, placeholder, prepare,
build, commands, stage_labels, session_rows) and the optional hooks (ADR P-05);
`resolve_host` / `load_spec` (from `agent.host`, `module:Class`); `memory.py` composes
governed long-term memory and dreams for any agent.

**`interface/tui`** — one Textual app (`AgenticTuiApp`, `python -m agent_platform.interface.tui`).
A live feed of collapsible stage cards, a status line above the input (the live stage, elapsed
time, tokens), slash commands with tab completion, side panels when a host offers them, the
Review screen (`ctrl+r`), and run controls (Stop → Resume / Abort). A card shows a field only if
it explains what happened, is not derivable from another field, and is read by something;
outcome colours only.

**`interface/commands`** — what an operator types. Commands return data (`CommandResult`),
never pixels, and never reach a model. `CommandRegistry` (`dispatch`, `completions`,
`argument_completions`) is the only place a slash is parsed.

| Command | Group | Owner | Does |
|---|---|---|---|
| `/help [command]` | Console | platform | Lists commands, or explains one |
| `/clear` | Console | platform | Clears the feed |
| `/audit [filters]` | Inspect | platform | The audit trail as a table, or one trace as a tree |
| `/stats` | Inspect | platform | This console session's numbers (turns, latency, tokens, checks), as a tree |
| `/review [list\|show\|approve\|deny]` | Inspect | platform | The human-review queue; a comment on every decision |
| `/memory` | Inspect | platform | Long-term memory as tables (scoped to a verified subject) |
| `/dream` | Inspect | platform | Consolidate pending dream events now |
| `/eval` | Evals | platform | Recent eval runs from the eval store |
| `/harness [path] [on\|off]` | Control | platform | Report or move the feature tree (hot reload) |
| `/resume`, `/abort` | Runs | platform (durable agents) | Continue or end a stopped run |
| `/skills [list\|show\|history]` | Agent | csa, ops | The agent's skills and their versions |
| `/void [order]` | Agent | csa | List issued refunds, or void one before it settles |
| `/cancel` | Agent | csa, ra | Stop the run in progress (checkpoint kept) |
| `/kb` | Agent | csa | The knowledge-base index |
| `/research [question]`, `/done`, `/waive <id\|all>`, `/approve` | Agent | ra | Conversational acts forwarded to the agent as turns |

**`interface/presentation`** — plain-text renderers shared by `make` output and the console:
`box_table`, `tree_lines` / `TreeNode`, `CommandView`, terminal colouring. Human-readable output
is a box-drawn table or tree.

## A25. evals

**Owns.** Three graders over an agent's suites, plus the intent bake-off (ADR P-28).

- **Layout.** `core/` (contracts — `EvalPlugin`, `GoldenCase`, `Transcript`, `Score`;
  fingerprint; `DuckDbEvalStore`; pipeline — generate, grade, gate, trials, orchestrator;
  reporting; cli), `oracle/` (`InvariantOracle`, `Rule`), `deepeval/` (the platform judge as a
  `DeepEvalBaseLLM`, `ConversationalTestCase` builder, metrics, grader, report), `promptfoo/`
  (runner over `npx promptfoo eval -c`, agent and grader providers, `oracle_assert`, report),
  `intent/` (accuracy, macro-F1, coverage, Brier, ECE, latency).
- **Seam.** `EvalPlugin` (agent_id, load_cases, prepare, agent_under_test, oracle,
  quality_metrics, conversation_frame, security_config, fingerprint_inputs), named in the
  agent's `evals.plugin`.
- **Caching.** Generation by (case hash, agent fingerprint), grading by (transcript hash,
  grader fingerprint). Ctrl+C marks the run interrupted; the next run resumes.
- **Store.** `db/agent_platform/evals.duckdb` — `eval_runs`, `eval_cases`, `eval_transcripts`,
  `eval_stages`, `eval_scores` and views (`v_run_summary`, `v_failures`, `v_metric_trend`, …).
- **Reports.** `reports/<package>/evals/<section>-<run_id>.md`, `run_id =
  <agent>-<suite>-<hash>-<YYYYMMDD-HHMMSS>`.
- DeepEval is an optional extra imported lazily; Promptfoo is a Node tool at a pinned version.
  Neither is a runtime dependency.
- **Config.** `evals:` — `plugin`, `suites`, `core`, `trials`, `store`, `fingerprint`, `gate`,
  `deepeval`, `promptfoo`, `reporting`.

## A26. analytics

**Owns.** The analytics source contract (ADR P-24, P-26). Schema detail is in
[analytics_schema.md](analytics_schema.md).

- **Semantic layer** (`semantic_layer.yaml` + agents' own layers, found by convention): each
  curated view's SQL and column meanings; the Tier-1 metric catalogue (`{view, expr, by}` plus
  `label`, `hint`, `group`).
- **Export** (`exporter.py`): renders every view to Parquet by atomic rename on the process
  holding the DuckDB lock (a running console every `analytics.export_interval_seconds`, or
  `make analytics-export`); fails if a view's columns drift from their documentation; free text
  masked by `analytics.redact_patterns`; writes `manifest.json`.
- **Sources.** `live` (`db/agent_platform/observability.duckdb` → `data/agent_platform/analytics/`)
  and `dummy` (`db/agent_platform/dummy_observability.duckdb` →
  `data/agent_platform/analytics_dummy/`, generated by `make dummy-data analytics` from every
  agent's `analytics/dummy.yaml`).
- `query_guard.py` (`QueryGuard`), `metric_sql.py`, `catalog.py`, `dashboards.py` (generates the
  Rill project under `analytics/dashboards/rill/`), `refresh.py`, `cli.py`
  (`python -m agent_platform.analytics export | dashboards | models | drop-legacy | dummy`).

---

# Part B — Agents

Each agent composes the platform the same way: `config/config.yaml` (its overlay),
`graph/builder.py::build_agent` (the composition root), `console/` (its `AgentHost`), and its
own domain packages. Agent subpackages often share a name with the platform package they plug
into (`gatekeeper/`, `memory/`, `review/`, …): that is where the agent's side of the seam lives.

## B1. Customer Support agent

`src/customer_support_agent/` — a refund agent that composes every platform feature.

| Package | Owns |
|---|---|
| `graph/` | `CustomerSupportAgent` (`handle`, `resume`), the wiring (`edges.py`), nodes (`nodes/`: authenticate, screening, intent, context, prefetch, propose, execute, finalize, review_wait), the composition root (`builder.py`) |
| `domain/` | `Order`, `User`, `OrderStatus`, `RefundStatus`; `Customer` identity (a context variable the model never sets); `OrderReferenceResolver` (the order under discussion); `RefundPolicy`, `CancellationPolicy` — the single source of business truth; `ports.py` |
| `deterministic_checks/` | `OrderChecks`: an explicitly named order exists, is the customer's, is not refunded, is refundable — before any model call |
| `intent/` | `CustomerSupportIntentRouter`: deterministic rejects → phrase rules → Laya (or BERT) at its own threshold; follow-ups ("what about 10010?"); the read-intent suggestion for an unplaced turn |
| `tools/` | Five typed `support` tools: `get_order_status`, `list_orders`, `get_refund_status` (reads), `process_refund`, `cancel_order` (writes); `CustomerSupportToolHandler` |
| `gatekeeper/` | `CustomerSupportBusinessRules` (the per-action refund limit from the persisted policy → RATIFIABLE) and `EstablishedOrderScope` (writes only target the order under discussion → else VOID) |
| `resilience/` | `RefundSaga`, `CancellationSaga`, `RefundSettlement` (issued → settles; `/void`) |
| `verification/` | `RefundEffectSource`: reads the refunds ledger and order back |
| `review/` | `ReviewQueueService` (approve resumes the waiting durable run); referral of escalated refunds |
| `console/` | `CustomerSupportHost` (prepare: warm the classifier and NLI, sync the knowledge base, recover orphaned runs; settlement sweep as `HousekeepingHost`); `/void`, `/cancel`, `/kb`, `/skills`, the support review handler (`refund`, `cancel_and_refund`, `reply`) |
| `loop_control/` | `OutputQualityGate`: `PlanLookup` (configured tool order per intent), `GroundedCritic` (one deterministic critique), at most one repair, `IndependentScorer` |
| `grounding/` | `GroundingEvidence` — every tier a reply is checked against, for one `TurnRecord` (order record, tool results and refusals, policy and prompt rules, articles, typed facts, the customer's words) |
| `presentation/` | Order menu and details rendered from the record; grounding facts (`facts.py`); stage-card fields |
| `prompts/` | `AGENT.md` (profile), `skills/*/SKILL.md` (order-status, order-details, list-orders, refund-workflow, refund-status, cancel-order), reply prompts, `customer_policy.md` |
| `memory/` | `support_memory_profile()` — session summaries by BART, checked by NLI; `TurnMemory` — what a finished turn leaves in working memory, the dream inbox and (as a claim) semantic memory |
| `persistence/` | `schema.sql`, `OrdersRepository`, `UsersRepository`, `RefundLedgerRepository`, `require_seeded` |
| `security/` | `CustomerAuthenticator` — seven-character user codes from an unambiguous alphabet |
| `observability/`, `analytics/` | `TurnAccountant` (SLO outcomes); Tier-2 facts, semantic layer and dummy profile |
| `knowledge_base/` | Informational articles retrieved as semantic memory; they never decide |
| `dummy_data/` | `generator.yaml`, seed SQL, `make dummy-data csa` (100 users, 1,000 orders, fixed seed) |
| `evals/` | `CsaEvalPlugin`, golden cases (50; 30 core), fixture (`EVAL001`–`EVAL005`, orders 90001+), oracle rules, DeepEval metrics, Promptfoo red team (27 attacks; 20 core), intent bake-off cases |
| `admin/` | `make refund-limit` |

**Turn flow.** See [HLD §5.1](HLD.md#51-customer-support) and §6.1. The order row the
deterministic checks read is reused by prefetch and the read path. Every turn is a durable
thread; a refund above the limit is queued and the run pauses in `await_approval`, while the
customer keeps chatting.

**Domain rules** (ADR C-01 – C-03): refundable = delivered, or cancelled before delivery;
already refunded = never again; full refund only; the direct limit is a database row; an issued
refund is voidable until it settles.

## B2. Research agent

`src/research_agent/` — a scoped, contract-bound research agent with a direct-answer path.

| Package | Owns |
|---|---|
| `graph/` | `ResearchAgent.handle` (dispatch on the session's phase); three graphs — `answer_graph.py` (reason ⇄ act → ground → finalize), `scoping_graph.py`, `run_graph.py`; their nodes; `builder.py` |
| `scoping/` | Research kind (`kind.py`: cues → example classifier → LLM), checklists per kind, hybrid termination (`termination.py`), the anti-sycophancy guard, time windows, `BriefBuilder` |
| `rubric/` | Templates per kind, the deterministic generator, the editor, freeze and hash |
| `planning/` | The planner and the plan validation gate (acyclic, bounded, traced) |
| `research/` | The worker (bounded act–observe per sub-question), the three content defences, extraction (verbatim spans), reflection (`GapAnalyzer`), stop rules (one per kind), recency; `RunLoopPolicy` (`loop_policy.py`) — stop after reflection, route after the audit, final status, from a `LoopSnapshot` |
| `synthesis/` | The report writer (its own outline, held to limits in `outline.py`), the citation guard, Coverage & Limits (built in code) |
| `audit/` | Named acceptance checks and `IndependentAuditor` (`audit_report(contract, report, ledger_snapshot)`) |
| `tools/` | Direct-answer tools: web search, page fetch, weather, clock — all through the platform tool path |
| `guardrails/` | The person's message screen (`input_screen.py`), page evidence for the direct answer |
| `memory/` | `research_memory_profile()`; `RunEventWriter` (run dream events) |
| `domain/` | `ScopeBrief`, `KnownUnknown`, `Rubric`, `Contract`, `Budget`, `Plan` |
| `console/` | `ResearchHost`: greeting, side panels (checklist, budget meters), cancel key, forwarded commands |
| `llm/`, `prompts/`, `context/`, `observability/`, `presentation/`, `analytics/` | Structured role calls; role instructions and `AGENT.md`; the direct answer's context window; telemetry; cards; Tier-2 facts |

**What each role may see.** Only the extractor reads raw page text. The planner, reflector and
synthesizer see the contract, the ledger's verbatim claims and run bookkeeping. The audit sees
exactly `(contract, report, ledger_snapshot)`.

**Budgets.** Searches, fetches, tokens, seconds, rounds, DAG nodes, gap-fill rounds,
concurrency — reserved synchronously before every await.

**Statuses.** COMPLETE (every must-pass item met, no budget exhausted), PARTIAL (anything less;
Coverage & Limits says why), CANCELLED, FAILED (plan invalid twice).

## B3. Ops agent

`src/ops_agent/` — the MCP demonstration agent.

| Package | Owns |
|---|---|
| `graph/` | `propose ⇄ execute → ground` (`graph.py`); `GovernedCalls` (VALID when allowed, VOID refused, UNGOVERNED when withheld); `builder.py` (`_sources` → `GovernedMcpToolSource`) |
| `guardrails/` | `FigureCheck` — every number in the answer must appear in a query result (level 2) |
| `presentation/` | The reply with its SQL and rows |
| `prompts/` | `AGENT.md` and one `always` skill, `analytics-investigation` (prefetches `list_metrics`, `list_views`) |
| `console/` | The ops host; `/skills` |
| `domain/`, `analytics/` | Its policy and declared outcomes |

---

# Part C — Cross-subsystem invariants

Tested at the integration level; owned by no single module.

| Invariant | Verified by |
|---|---|
| `agent_platform` names no specific agent, in code, config or docs | `tests/agent_platform/test_platform_boundary.py`, `test_roles_and_boundary.py` |
| No module sits loose at a package root | `tests/agent_platform/test_package_layout.py` |
| No agent subpackage imports another in a cycle; domain never imports persistence | `tests/agent_platform/test_agent_layering.py` |
| Every model-proposed call takes the one tool path; the Executor runs only approved (or audited UNGOVERNED) calls | `tests/integration/test_cross_subsystem_invariants.py`, `tests/customer_support_agent/test_gatekeeper.py` |
| No write executes outside the established target's scope | `test_gatekeeper.py::TestWritesAreScopedToTheEstablishedOrder` |
| Attacks against the harness are defended | `tests/integration/test_e12_attacks.py`, the Promptfoo suite |
| No side-effecting result is authoritative without read-back | refund flow tests; verification owed dedicated unit tests |
| Every retry of a side effect is idempotent | `tests/agent_platform/test_resilience.py` |
| A Protocol has a real implementation | `mypy --strict` |
| No YAML key that no settings model declares | `tests/customer_support_agent/test_config_surface.py` |
| Tests never call a provider | `tests/conftest.py::no_provider_calls` |
| Coverage ≥ 80% | `pytest-cov` `fail_under` (`make verify`) |
