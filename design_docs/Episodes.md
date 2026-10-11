# The Agentic AI Systems Masterclass — v10 (Complete, Self-Contained)

This document is the full script-feed for all 48 episodes. Every episode below is stated in full — Principles Applied, Key Concepts, Key Points, Exit Artifact — so this file can be used standalone to write each film-ready script.

Series-wide meta-framework: the Series Contract, the teaching loop (Live Failure → Architectural Question → Borrowed Principle → Formal Model → Reference Implementation → Adversarial Test → Measurable Invariant → Commit), the Reference Architecture diagram, the ten Series-Wide Design Rules, and the closing mantra.

---

## Module 0 — Foundations

### E01 — Agent = Model + Harness

**Principles Applied:**

- **Agency Theory (economics/law):** a principal delegates a goal to an agent who reasons on the principal's behalf; delegation of reasoning is not delegation of authority. This is the seed of the entire series.
- **Design by Contract (Meyer, 1986), previewed:** every interface the harness exposes to the model — a `ToolCall`, an `AuthorityDecision` — should state its guarantees explicitly, the way a function signature states its pre/postconditions, rather than being implied by convention.

**Key Concepts:** Model (stateless reasoning component) · Harness (the surrounding control system) · Agent = Model + Harness · Proposal vs. Execution.

**Key Points:**

- Cold open: a naked model receives and executes a $4,000 refund request unsupervised; a small external authority check is added; identical model, identical prompt, different outcome.
- Establishes the design invariant: the model may propose, only the harness decides whether a proposal reaches execution.
- Exit artifact: a minimal typed `ToolCall` / `AuthorityDecision` / execution interface, committed to `agentic-harness-core` v0.

### E02 — Agency ≠ Authority

**Principles Applied:**

- **Agency Theory — actual vs. apparent authority (Restatement of Agency):** an agent claiming authority, however convincingly, does not create real authority; only a principal further up the hierarchy can grant it.
- **Jensen & Meckling (1976), agency cost:** the gap between what an agent does and what a principal actually wants is a cost that must be bounded by design, not trusted away.
- **Saltzer & Schroeder — Separation of Privilege (1975):** a consequential action should require more than one independent condition to be satisfied, not a single check that a clever actor can satisfy alone.

**Key Concepts:** Agency · Authority · Authority Hierarchy · Structuring attack · Cumulative ledger.

**Key Points:**

- Cold open: 80 × $50 refunds (each individually authorized) sum to an unauthorized $4,000 outcome — a structuring attack.
- Second attack: untrusted ticket text claims false third-party pre-approval; shown to fail because untrusted content sits outside the Authority Hierarchy entirely.
- Exit artifact: Authority Hierarchy (provider > org > product > workspace > user task > runtime reminders > tool observations > untrusted content, where only ranks 1–5 can grant/expand authority) + per-action authorization check + cumulative-spend ledger.

### E03 — Autonomy Is a Control Allocation, Not a Feature Toggle

**Principles Applied:**

- **Sheridan & Verplank's Ten Levels of Automation (1978):** a 1–10 scale from "human does everything" to "computer decides and ignores the human" — automation is a graduated allocation of control, never a binary "is it AI or not."
- **Kaplow's Rules vs. Standards (1992):** a law-and-economics result on when to pre-commit to a fixed rule (fast, predictable, cheap to apply, but brittle at the edges) versus leave a decision to case-by-case judgment (flexible, but expensive and unpredictable to adjudicate) — directly informs whether a given agent action should be a deterministic rule or a judged decision.
- **Ashby's Law of Requisite Variety (cybernetics, 1956):** a controller can only regulate as much variety (unpredictability) as it itself possesses; if the environment is more variable than the harness's control/verification capacity, autonomy has outrun control.

**Key Concepts:** Path Cardinality (how many genuinely different valid paths exist through a task) · Blast Radius (scope of damage a wrong action can cause) · Certifiability (how cheaply and independently success can be verified) · **The Decision Matrix — Three Gates**, combining the three variables into a single scorable framework · Automation level (Sheridan's 1–10 scale, applied per action class, not per system).

**Key Points:**

- The Three Gates replace the naive "workflow vs. agent" debate with a scored decision: high path cardinality + low blast radius + strong certifiability clears all three gates toward high autonomy; low cardinality or high blast radius fails a gate and routes toward deterministic policy or mandatory human approval.
- Sheridan's scale is overlaid on the Three Gates output, producing an automation level per action class, not a single system-wide autonomy setting.
- Running demo: the same refund process assigned three different automation levels for three different action classes within one system.
- Exit artifact: an action-level automation policy table (e.g., read order → high; <$50 refund → high; $50–$500 → medium/logged; >$500 → Level 5, human approval required).

### E04 — Workflow, Agent, or Hybrid?

**Principles Applied:**

- **Structured Programming (Dijkstra, and the classical control-flow vocabulary):** sequence, branch, and loop as the deterministic building blocks every workflow pattern is composed from.
- **Control Theory, previewed:** feedback as the vocabulary underlying every agentic (autonomous) pattern — fully developed later in M6/E20.
- Reuses E03's Three Gates as the selection mechanism, applied live rather than re-derived.
- **CMMI — Capability Maturity Model Integration (SEI/Carnegie Mellon, 1991), previewed:** CMMI's core claim — that a staged maturity model only works if each level's practices are _evidenced infrastructure_ for the level above it, not a self-declared label — is introduced here as the reason the Three Gates score, not a vendor's "autonomous" marketing claim, is what should ever move an action up Sheridan's scale. This plants the question the whole series answers architecturally, module by module, and is paid off explicitly in **M16/E37**, where the reference harness's own maturity is scored against actual SLO/audit evidence rather than asserted.

**Key Concepts:** WF patterns — sequence, branch, fixed loop, fan-out/fan-in, map-reduce, deterministic router, human-in-the-loop, RAG pipeline · AG patterns — tool use, ReAct, planning/decomposition, orchestrator-subagent, evaluator-optimizer, reflection, agentic retrieval, and _memory systems (named only here — full treatment in M5; individual pattern depth for the others in M1, so nothing is taught twice)_.

**Key Points:**

- Architect's rule stated once and reused for the rest of the series: put determinism around consequences, autonomy around uncertainty.
- Live design exercise: three real requirements — a fixed fraud rule, an ambiguous customer explanation, a novel refund exception — resolved into **one hybrid system**, not three disconnected demos.
- Exit artifact: a pattern-selection decision record showing which Three-Gates score drove which pattern choice, extended with a column mapping each AG pattern to the M1 episode that gives it full depth.

---

## Module 1 — Agentic Design Patterns 

### E05 — Tool Use & ReAct: Reason → Act → Observe as the Foundational Loop

**Principles Applied:**

- **Newell & Simon's Physical Symbol System / means-ends analysis (classical AI planning):** the theoretical ancestor of treating problem-solving as a search through a space of actions toward a goal state — what ReAct formalizes for language agents.
- **Yao et al.'s ReAct (2022):** the empirical result that interleaving explicit reasoning traces with actions and observations outperforms either reasoning-only (chain-of-thought) or acting-only approaches on multi-step tasks.

**Key Concepts:** Thought → Action → Observation cycle · Tool call as an atomic unit · Trajectory (the accumulated sequence of thought/action/observation triples).

**Key Points:**

- Live demo: the same refund-lookup task run as a single unstructured completion (fails silently on a multi-step lookup) vs. as an explicit ReAct loop (succeeds, with the reasoning trace visible for audit).
- Establishes the ReAct loop as the atomic unit every later pattern in this module composes from — planning sequences multiple ReAct cycles, reflection critiques a completed trajectory, orchestration runs multiple trajectories in parallel.
- Where it fails: long-horizon tasks where the trajectory grows past useful context (forward reference to M5/E17, Context Failure) and tasks with no natural stopping point (forward reference to M6/E20, Loop Engineering) — the motivating gaps for the rest of this module and for M6.
- Exit artifact: a minimal `react_step()` function added to `agentic-harness-core`, composed from E01's `ToolCall` interface.

### E06 — Planning & Decomposition, Reflection, Evaluator-Optimizer

**Principles Applied:**

- **Hierarchical Task Network planning (classical AI planning):** decomposing a high-level goal into an ordered set of sub-goals, each independently achievable, before execution begins.
- **Shinn et al.'s Reflexion (2023):** verbal self-critique, stored and reused across attempts, as a substitute for gradient-based learning when weights can't be updated.
- **The Generate-Test-and-Debug loop (general problem-solving methodology):** the ancestor of Evaluator-Optimizer — produce a candidate, test it against a criterion, revise, repeat.

**Key Concepts:** Plan (an explicit, ordered decomposition committed before execution) · Reflection (post-hoc verbal self-critique fed back as context) · Evaluator-Optimizer loop (a generator paired with a separate, independent scorer).

**Key Points:**

- Decomposition is framed as recursively applying E04's WF-vs-AG decision to each sub-task — a plan is not one autonomy decision, it's a tree of them.
- Live demo: a multi-step refund-and-goodwill-credit task decomposed into an explicit plan, executed step by step, with each step independently scored.
- Reflection's failure mode is demonstrated directly: an agent reflects on a wrong answer and produces a more confident, still-wrong answer, with no external grounding to catch it — a direct forward reference to M11 (Hallucination Protection), which is what actually fixes this.
- Evaluator-Optimizer is shown explicitly reusing M3's Eval Pyramid components as its optimizer signal rather than inventing a fresh ad hoc metric — the same discipline, applied inside the loop instead of after it.
- Exit artifact: a `Planner` + `Reflector` + `Evaluator` composition added to the harness, each independently swappable per Parnas's Information Hiding (previewed here, fully stated in M2/E08).

### E07 — Orchestrator-Subagent & Agentic Retrieval

**Principles Applied:**

- **Divide-and-conquer / recursive decomposition (classical algorithms):** breaking a large problem into independently solvable sub-problems whose results are recombined — the direct ancestor of orchestrator-subagent.
- **Information Retrieval as an active process:** the distinction between agentic retrieval (the system decides when and what to retrieve, as part of its reasoning) and classical RAG (retrieve-then-generate, fixed at pipeline design time).

**Key Concepts:** Orchestrator (owns the plan and delegates) · Subagent (owns a bounded sub-task, no visibility into the parent's full context) · Agentic retrieval (retrieval as a tool call the model chooses to make, not a pipeline stage).

**Key Points:**

- Orchestrator-subagent is taught here at pattern-shape level only — what it is and when it's the right decomposition — with its full economics, topology choices (supervisor/hierarchy/peer/swarm), and coordination cost built out in **M17 (Multi-Agent Architecture)**, so the pattern isn't re-explained, only its cost model is added later.
- Agentic retrieval is taught here at pattern-shape level, with its full retrieval mechanics (chunking, hybrid search, reranking, graphs) built out in **M5/E19 (Production RAG)**.
- Memory-systems-as-a-pattern is named a second time and pointed at M5 — still not re-taught, per the rule E04 already established.
- Exit artifact: the pattern-selection table from E04, now extended with a column mapping every AG pattern to the specific episode that gives it full depth — the map the rest of the series hangs off.

---

## Module 2 — Harness Engineering

### E08 — The Harness Contract: Six Subsystems, Explicit Guarantees

**Principles Applied:**

- **Design by Contract (Meyer, 1986), fully developed:** every subsystem states explicit preconditions, postconditions, invariants, and — critically — an explicit "not guaranteed" list, the artifact a CISO can be handed instead of a verbal walkthrough.
- **Parnas's Information Hiding / Separation of Concerns (1972):** each of the six subsystems owns one responsibility and exposes a narrow interface, so a change inside one subsystem shouldn't ripple through the others.
- **ETL / Compiler Pipeline design (general software engineering):** context assembly specifically is a fixed sequence of independently-testable stages, not an ad hoc string concatenation.

**Key Concepts:** The six subsystems — context assembly, output parsing, tool execution, control loop, state management, error handling · The five-stage context-assembly pipeline — `gather_candidates → score_relevance → fit_to_budget → format → order`.

**Key Points:**

- For each of the six subsystems, the episode defines preconditions, postconditions, invariants, failure semantics, and observability requirements before a single line of any later module's deeper content is taught.
- Context assembly's mechanics are built out fully here as part of the contract; tool _selection_ mechanics (the registry, namespacing, retrieval-over-tool-descriptions problem) are named as belonging to the tool-execution subsystem but deferred whole to **M8/E22**, so they get their own full episode rather than being split thin across two places.
- Exit artifact: the full architecture contract for `agentic-harness-core`, the reference every subsequent module extends. Exit metrics for context assembly specifically: token budget, tool-selection accuracy, irrelevant-tool exposure rate, context assembly latency.

---

## Module 3 — Evals

### E09 — The Eval Pyramid: From Unit Invariant to Production Shadow

**Principles Applied:**

- **The Software Testing Pyramid (Mike Cohn, 2009):** unit/integration/E2E/production-monitoring, mapped directly onto the agent-eval equivalent — a discipline the audience already trusts, deliberately borrowed rather than reinvented.
- **Scientific Reproducibility / Experimental Design (general methodology):** an eval is a repeatable measurement against an explicit, falsifiable claim — not "asking the model 100 questions and eyeballing the answers."

**Key Concepts:** Component eval · Integration eval · End-to-end golden-task eval · Production shadow eval.

**Key Points:**

- Four levels, narrower and more expensive moving up the pyramid.
- Every component gets an explicit contract, input distribution, expected properties, scorer, and regression threshold.
- Exit artifact: an eval manifest checked directly into the repository, not living in a spreadsheet or a person's head.

### E10 — NOLLM First: The Eval Cascade and the Cost of Measurement

**Principles Applied:**

- **Cascade Classifier Design (Viola & Jones, 2001):** run cheap, fast rejectors first and escalate to expensive, precise classifiers only when the cheap stage can't resolve the case — the direct architectural ancestor of the NOLLM scoring stack.
- **Signal Detection Theory (previewed, fully developed in M11/E25):** every scorer in the cascade is ultimately a threshold-selection problem trading off false positives against false negatives.

**Key Concepts:** NOLLM Scoring Stack — exact assertion → deterministic rules → embedding similarity → NLI entailment → calibrated LLM judge → human adjudication · `nollm_scorer_ratio` — the fraction of verdicts resolved by the cheap stages vs. escalated, tracked as a series-wide cost-discipline metric.

**Key Points:**

- If every eval requires an LLM judge, measurement itself becomes expensive, slow, noisy, and impossible to run continuously — the exact failure this cascade prevents.
- This same cascade logic is reused twice more later in the series in genuinely different roles: as a _retrieval_ tier ordering in M5/E14 (cheap exact/regex lookup before embedding similarity), and as a _cost_ tier ordering in M18/E44 (cheapest sufficient model tier first) — flagged here as a preview so the pattern is recognized, not re-derived, each time.
- Exit artifact: a reusable scoring service that every later module's eval segment calls into rather than reimplementing.

### E11 — Evaluation Integrity: Poisoned Labels, LLM Judges, Goodhart & Human Agreement

**Principles Applied:**

- **Thurstone's Law of Comparative Judgment (1927):** pairwise comparison (A vs. B) correlates with human judgment far better than absolute rating (score A from 1–5) — psychometrics, not AI-industry folklore.
- **Cohen's Kappa (1960):** the statistic that distinguishes genuine rater agreement from agreement that would occur by chance alone; the calibration bar for any judge, human or LLM.
- **Goodhart's Law (1975, popularly restated by Marilyn Strathern):** when a measure becomes a target, it ceases to be a good measure — an agent optimized purely for pass rate, satisfaction score, or "task completed" will eventually find a way to satisfy the metric while degrading the actual objective.
- **Data-quality auditing / label-bias correction (general statistical methodology):** the discipline for correcting an eval or training set that reflects historical practice rather than written policy.

**Key Concepts:** Poisoned ground truth · Three-bucket relabeling framework · Pairwise comparison · Judge calibration · Metric gaming.

**Key Points:**

- Live experiment: a behavior is tuned against a single weak metric, then shown to degrade under an unseen rubric — Goodhart's Law demonstrated, not just asserted.
- Two baselines reported honestly side by side (human-vs-written-policy, human-vs-amended-policy) instead of picking whichever flatters the system.
- Exit artifact: golden dataset provenance ledger + judge calibration report.

---

## Module 4 — Gatekeeper Architecture, Design & Implementation

### E12 — The Agent Control Plane: Sandbox → Gatekeeper → Execution

**Principles Applied:**

- **Reference Monitor concept (classical computer security):** every security-relevant request must pass through a single, always-invoked, tamper-proof checkpoint — the formal justification for "no execution path bypasses the Gatekeeper."
- **Saltzer & Schroeder's Complete Mediation:** every single access to every protected resource must be checked, not just the first or the obvious ones — no caching a "yes" from a prior check.
- **Saltzer & Schroeder's Principle of Least Privilege:** every permission held should be the minimum necessary for the current task, for the minimum necessary time — implemented here as role-based permissions, task-scoped grants, and just-in-time elevation.
- **Saltzer & Schroeder's Fail-Safe Defaults:** the default answer to an unrecognized or ambiguous request is deny, not allow.
- **Zero Trust (NIST SP 800-207):** never grant trust merely because a caller is internal, previously approved, or "the agent" — authority is re-verified at every security-sensitive crossing, not assumed to carry over.
- **Segregation of Duties (trading-desk compliance regulation):** the same principle that prevents one person from both initiating and approving a trade, applied so that proposing an action and authorizing it are structurally different roles.
- **The Semantic Trap:** if the Gatekeeper itself ever consults an LLM for judgment, that judgment must be Tier-2 — advisory or routing only — never given veto power over the authority boundary a deterministic Tier-1 check already governs. A model may flag a request as suspicious; it may not redefine whether the request is authorized.

**Key Concepts:** Sandbox (isolated proposal workspace) · Gatekeeper (the single mediation point) · Execution (irreversible real-world effect) · Four Surface Classes — Locked, Editable, Append-only, Human-controlled · Tier-1 (deterministic, authoritative, can block) vs. Tier-2 (semantic, advisory-only).

**Key Points:**

- The Gatekeeper answers four independent questions on every request: what surface is this, who is asking, what authority was granted, has cumulative state crossed a threshold.
- Live attacks, run one after another: direct tool execution attempt bypassing the sandbox; prompt injection requesting a privileged write; the agent attempting to edit its own policy; cumulative refunds crossing the E02 ledger threshold.
- Live demonstration of the Semantic Trap itself: a Tier-2 LLM flags a technically-permitted but semantically-suspicious refund; the system shows flag → escalate → continue as the correct path, and shows what breaks if Tier-2 is wrongly given block authority.
- Exit invariant: no execution path exists that bypasses the Gatekeeper, and no role, however senior, gets write access to a Locked surface regardless of its other permissions.

---

## Module 5 — Memory: Context Management, Episodic, Semantic, Procedural

### E13 — Memory Architecture: Working, Semantic, Episodic, Procedural

**Principles Applied:**

- **CoALA — Cognitive Architectures for Language Agents (Sumers et al., 2023):** the formal four-store framework this episode is built on.
- **Park et al. (2023), generative-agents retrieval scoring:** a composite retrieval signal — `relevance + recency + importance` — rather than any single signal, extended here with authority and temporal validity for production use.
- **Classical Cognitive Architecture theory (the Soar/ACT-R lineage), briefly:** the working/long-term memory split long predates language agents; CoALA is a recent, LLM-specific instantiation of a much older idea.

**Key Concepts:** Working memory (what's active now — literally the context window) · Semantic memory (what's generally true) · Episodic memory (what happened) · Procedural memory (how to do things).

**Key Points:**

- Central rule: memory type is a semantic decision about the fact's nature, not a storage-engine choice made for convenience.
- Explicitly reconnects to E04's agentic-pattern catalogue: this is the full treatment of the "memory systems" pattern that was only named there.
- Composite retrieval formula taught with worked examples showing why single-signal retrieval (pure recency or pure similarity) feels subtly wrong.

### E14 — Building Each Store: Core Retrieval Logic **[NEW]**

**Principles Applied:**

- **Unix philosophy / "do one thing well" (McIlroy):** grep, ripgrep, and plain regex matching as legitimate, composable retrieval primitives in their own right, not a fallback for when "real" retrieval is unavailable.
- **The NOLLM Scoring Stack, reused from M3/E10, reapplied as a retrieval stack:** cheap, deterministic, auditable lookup (exact match, regex) attempted before expensive, fuzzy lookup (embedding similarity), the same cascade logic applied to _retrieval_ instead of _scoring_.

**Key Concepts:** Structured fact store (semantic memory, exact/fuzzy key lookup) · Append-only event log (episodic memory, filesystem-based, one file per session or day, ripgrep-able by construction) · Versioned how-to artifact (procedural memory) · Working memory as a pointer, not a store (deferred to E15–E17, not duplicated here).

**Key Points:**

- Semantic memory built live as a structured fact store, with **regex/grep/ripgrep-based retrieval taught explicitly** as the cheap, deterministic, first-tier lookup for well-structured facts — reached for before any embedding model, per the reused NOLLM cascade discipline.
- Episodic memory built live as a filesystem-based, append-only log (one file per session), queried by time range and participant using the same ripgrep-first discipline — shown as the deployable-Monday version before a database-backed version is introduced.
- Procedural memory built live as versioned "how-to" artifacts capturing corrected mistakes and tool-use recipes, explicitly linked forward to **M7/E21**, where a DSPy-compiled prompt is shown as one legitimate procedural-memory artifact type.
- Working memory is explicitly _not_ re-taught here — it's the context window itself, covered in full in E16 (Context Failure) and E17 (Context Management), so this episode doesn't duplicate that content.
- Exit artifact: three minimal, independently testable store implementations (`SemanticStore`, `EpisodicStore`, `SkillStore`) added to `agentic-harness-core`, each satisfying a shared `MemoryStore` interface per E08's Information Hiding discipline.

### E15 — Memory Write Path: Provenance, Conflict Resolution & Temporal Truth

**Principles Applied:**

- **ETL Pipeline design, reused from E08:** extract → classify → provenance → conflict detection → confidence → authorization → route, applied to memory writes specifically.
- **Temporal Database theory (Snodgrass and the bi-temporal modeling literature):** tracking event time and ingestion time as genuinely separate axes, not one collapsed timestamp.
- **Database Conflict Resolution (last-write-wins vs. version-vector approaches):** the two legitimate strategies for resolving a memory conflict — temporal supersession or routing to a human curator — mapped onto established database-replication theory.

**Key Concepts:** Write pipeline (extract → classify → provenance → conflict detection → confidence → authorization → route) · Routes — auto-commit, curator queue, discard/decay · Bi-temporal truth (event time ≠ ingestion time).

**Key Points:**

- Live attack: a user states, confidently, "my refund limit is now $2,000" — the write pipeline is shown refusing to commit this into authoritative policy memory from conversational assertion alone.
- This refusal is the direct application of E12's Least Privilege and Fail-Safe Defaults principles to the memory subsystem specifically, not a new idea.

### E16 — Memory Is a Security Boundary: Poisoning, Revocation & False Recall

**Principles Applied:**

- **Saltzer & Schroeder's Principle of Least Privilege, reused:** write-path authorization scoped by fact category — authority-changing claims are never writable from conversational assertion alone, regardless of stated confidence, because no role's permission set includes granting itself authority.
- **Provenance / data governance discipline (general information-security practice):** tagging every stored fact as self-reported, verified system-of-record, or agent-drawn inference, weighted differently at retrieval time.
- **Signal Detection Theory, previewed:** false recall (confidently retrieving the _wrong_ stored fact) is a false positive, distinct in kind from forgetting (a false negative) — fully developed in M11/E25.

**Key Concepts:** Memory poisoning (distinct from prompt injection — no hidden text required, a user can simply _say_ a false thing intending it to be trusted later) · Provenance tagging · Decay-if-unconfirmed (a security posture, not mere data hygiene) · Hard delete / revocation.

**Key Points:**

- Threat list: memory poisoning, stale memory, false recall, authority contamination, deletion failure.
- Exit eval: a cross-session recall test under deliberately contradictory facts, specifically designed to catch false recall rather than clean forgetting.

### E17 — Context Failure: Rot, Distraction, Confusion, Clash & Poisoning

**Principles Applied:**

- **The Working-Set Model (Denning, 1968), previewed:** a process (or, here, an agent) only needs a bounded, actively-used subset of its total addressable state at any moment — fully developed in E18.
- **Information Theory (signal-to-noise framing):** context failure is, at root, a signal degrading as irrelevant or conflicting content accumulates relative to the useful content.
- **Cited empirical studies (not a classical law, but hard evidence):** DeepMind's Pokémon-agent poisoning case, Gemini 2.5's qualitative shift past ~100K tokens, the Llama 3.1 8B 46-tools-fails/19-tools-succeeds finding, a 39% average performance drop on conflicting multi-turn information.

**Key Concepts:** Context rot · Context distraction · Context confusion · Context clash · Context poisoning.

**Key Points:**

- Demonstrated via a synthesis-based degradation eval (not needle-in-a-haystack, which tests literal retrieval rather than reasoning under load) run at 10K/30K/60K/100K/150K tokens.
- Exit metric: maximum usable context at target task SLO — the harness's real operating budget, not the advertised window size.

### E18 — Context Management & Memory Consolidation: Offload, Reduce, Retrieve, Isolate, Cache, Consolidate

**Principles Applied:**

- **The Working-Set Model (Denning, 1968), fully developed:** the direct mapping this episode is built on — offload/reduce/retrieve/isolate/cache correspond almost exactly to swap/compress/page-in/partition/cache in classical virtual memory management.
- **Cache Locality of Reference (general systems theory):** why caching strategy depends on how predictably the same content recurs.
- **[NEW] Reflection-based memory consolidation, extending Park et al. (already cited in E13):** the same "reflection" step generative agents use to compress raw observations into higher-level insights, applied here as a sixth, offline context-management strategy — run periodically across sessions rather than within a single context window. Framed strictly as an engineering analogy to sleep-dependent memory consolidation (hippocampus-to-cortex transfer), not a claim about how the system internally works.

**Key Concepts:** Offload (must be restorable) · Reduce (compaction — percentage-based, fixed-token, and predictive strategies compared) · Retrieve (Tool RAG) · Isolate (per-agent context) · Cache (semantic and exact-match caching) · **Consolidate (new) — an offline batch process compressing raw episodic entries from E14 into durable semantic/procedural facts.**

**Key Points:**

- Central principle stated once and reused: context is a working set, not a landfill.
- Live cache-mechanics demo: explicit cache-control breakpoints vs. automatic prefix caching, with a live hit-rate comparison driven by deterministic vs. non-deterministic serialization.
- Live consolidation demo **[NEW]**: a week of raw episodic entries from E14's filesystem store consolidated into a handful of semantic facts, shown side by side with the retrieval-quality and token-cost difference against the unconsolidated version — explicitly framed as Reduce's cross-session analogue, keeping the E14 filesystem stores from growing unbounded.

### E19 — Production RAG: Chunking, Hybrid Retrieval, Reranking & Graphs

**Principles Applied:**

- **The Vector Space Model (Salton, 1975):** the mathematical grounding for similarity-based dense retrieval.
- **Library Science / Classification theory (the Dewey Decimal System, 1876, as intuition):** the older discipline of structured classification and retrieval that hybrid search and metadata filtering are a modern instance of.
- **Graph Theory, applied to GraphRAG:** relationship-dependent queries are answered by traversal, not by similarity, and the episode teaches when that distinction matters.

**Key Concepts:** Semantic chunking · Hybrid search (lexical + dense) · Cross-encoder reranking · GraphRAG.

**Key Points:**

- Decision rule taught explicitly: identifier-shaped queries favor lexical strength; relationship-shaped queries favor graph traversal; narrative semantic queries favor dense retrieval — production systems combine these rather than committing to one retrieval primitive.
- Vector database selection rubric compared across Pinecone, Weaviate, Chroma, and pgvector on metadata filtering, hybrid-search support, and deletion semantics — the last explicitly tied back to E16's revocation requirement.
- This episode is the full build-out of the "agentic retrieval" pattern-shape introduced in M1/E07.

---

## Module 6 — Loop Engineering & Loop Detection

### E20 — Designing Agent Loops: Stability, Deadlock, Livelock & Watchdogs

**Principles Applied:**

- **Control Theory / Cybernetics (Wiener, 1948):** an agentic loop is a feedback controller — observe, decide, act, observe result, update state, repeat — and it deserves to keep running only if it has a measurable goal, feedback, bounded state, a stopping criterion, and a resource budget.
- **Operating-System Deadlock and Livelock Theory (Coffman et al., 1971):** the classical taxonomy of ways concurrent or iterative processes stop making progress — deadlock (frozen), livelock (busy but stuck), oscillation — applied directly to agent loops that appear active but aren't converging.

**Key Concepts:** Objective, feedback, progress measure, termination condition, resource budget (the six things a legitimate loop must have) · Failure taxonomy — repetition, semantic stagnation, oscillation, deadlock, livelock, goal drift, cost runaway.

**Key Points:**

- Six-layer watchdog stack: step count, wall-clock, token budget, monetary budget, duplicate-action detection, semantic-stagnation detection — each demonstrated killing an induced loop at its own layer.
- Explicitly closes the failure mode M1/E05 flagged (ReAct loops with no natural stopping point).
- Exit artifact: a reusable `LoopController` component wired into the running harness.

---

## Module 7 — Prompt Engineering & DSPy

### E21 — Prompt Engineering as Software: From Hand-Tuned Strings to DSPy-Compiled Programs

**Principles Applied:**

- **Declarative Programming / Compiler Theory (the SQL/Prolog lineage):** separate _what_ the program should produce (a typed signature: given these inputs, produce this output, satisfying this metric) from _how_ it's phrased to the model — the model's exact wording becomes a compiled artifact, not hand-written prose.
- **Empirical, Gradient-Free Optimization (general ML methodology):** DSPy's "teleprompters" search the space of prompt formulations and few-shot examples against a metric, the same way a hyperparameter search optimizes a model.
- **"Software 2.0" framing (Karpathy):** once a component's exact text is chosen by an optimizer against a measurable objective rather than typed by a human, it behaves like a compiled artifact — versioned, regression-tested, and reproducible — not like content.

**Key Concepts:** Signature (a typed input → output contract, independent of wording) · Module (a composable DSPy program unit, e.g., `ChainOfThought`, `ReAct`) · Optimizer / Teleprompter (the search procedure that compiles a signature + metric into a concrete prompt and example set) · Prompt as compiled artifact vs. prompt as hand-written string.

**Key Points:**

- Opens with a short recap bridging the two prerequisites this episode actually depends on even though it's viewed at pillar position 7: a typed I/O contract (fully taught in M10/E24) and a scorer to optimize against (fully taught in M3) — both already covered by this point in the series.
- Opens by retiring "prompt engineering" as a standalone folk craft: a hand-tuned prompt string is untested, unversioned, and silently breaks on model upgrades — exactly the failure class M2/E08's Design by Contract was introduced to prevent.
- Live demo: the same task implemented first as a hand-written prompt, then as a DSPy signature compiled against one of M3's evals — compares reliability and regression behavior across a model swap.
- Explicitly scoped: DSPy optimizes the _wording and examples_ within a fixed signature; it does not replace the Gatekeeper, the schema validation from M10/E24, or any authority check — those remain outside the model entirely, per E02's central rule.
- Exit artifact: one production prompt in the Refund Agent re-implemented as a compiled, versioned DSPy module with its own eval-backed regression test — later reused in M5/E14 as an example procedural-memory artifact, and in M16/E39 as an example of a versioned artifact tracked through deployment.

---

## Module 8 — Tool Calling & Tool Registry

### E22 — Tool Selection at Scale: Registry, Namespacing & Retrieval-Based Selection

**Principles Applied:**

- **Information Retrieval, fully developed:** tool selection at scale is a retrieval problem, not a "list everything and hope" problem.
- **Reuses M5/E14's NOLLM-first retrieval-tier logic** (exact match → metadata filter → embedding similarity), applied here specifically to tool descriptions instead of memory facts.

**Key Concepts:** Tool namespaces · Retrieval over tool descriptions · Metadata-based filtering · Schema minimization · Tool registry problem (more tools is not inherently better than fewer, well-selected ones).

**Key Points:**

- Deferred whole from M2/E08, which names tool execution as one of the harness's six subsystems but defers the selection mechanics here for full treatment.
- Live demo: a 50-tool registry degrading task accuracy when exposed unfiltered, recovered by namespacing and retrieval-based filtering.
- Exit metrics: token budget, tool-selection accuracy, irrelevant-tool exposure rate, context assembly latency — the same four metrics named in E08, now measured against the built solution.

---

## Module 9 — State Management & FSM

### E23 — State, Events & Durable Execution

**Principles Applied:**

- **Automata Theory (Moore and Mealy machines, 1950s):** state is a distinct, named mode of execution, and transitions between states are either legal or illegal — nothing is inferred from scattered flags.
- **Write-Ahead Logging / the ARIES recovery algorithm (database engineering):** durable, replayable state is the classical answer to "what survives a crash."
- **Event Sourcing (Fowler):** a checkpoint is properly a serialized `(state, transition history)` pair, not an arbitrary snapshot — this is what makes "resume" well-defined rather than best-effort.
- **Lamport's Happened-Before Relation (1978):** when multiple workers act concurrently, the question that matters for audit and debugging isn't "what is the timestamp" but "what causally happened before what."

**Key Concepts:** Explicit state table (IDLE → PLANNING → ACTING → WAITING_TOOL → WAITING_HUMAN → DONE → FAILED) · Legal transition · Checkpoint as `(state, transition history, durable facts)` · Causal ordering.

**Key Points:**

- Live demo: a crash is injected after a refund side effect but before the next state write; recovery is shown to restore a _legal_ state without duplicating the refund.
- Lamport ordering introduced specifically for the multi-worker concurrent case, previewing M17's multi-agent coordination problems.
- Exit invariant: recovery can only restore a legal state, never an inferred or ambiguous one.
- This episode's Circuit Breaker FSM callback (Closed → Open → Half-Open, taught fully in M15/E35) is flagged here as the canonical example of this module's own state-table discipline applied to a dependency instead of the agent itself.

---

## Module 10 — Output Parsing: Pydantic, LMQL, Outlines

### E24 — Structured Outputs, Parsing & Fail-Loud Interfaces

**Principles Applied:**

- **Formal Language Theory (Chomsky, 1956):** constrained decoding enforces a grammar at the token level during generation itself — a hard structural guarantee, distinct from validating free text after the fact.
- **The Fail-Fast Engineering Principle:** a malformed output should halt and flag immediately, not be silently coerced into something plausible-looking and wrong.
- **Design by Contract, reused from M2/E08:** the parser is framed explicitly as part of the security and reliability boundary, not mere formatting glue.

**Key Concepts:** Free-form text vs. schema-constrained output vs. constrained decoding · Syntactic validity vs. semantic correctness vs. executable validity — three genuinely different properties, easily conflated.

**Key Points:**

- Live demo: a malformed tool call handled four different ways — silently coerced, rejected, automatically repaired, deterministically represented — with the tradeoffs of each made explicit.
- **[NEW]** The four handling strategies are mapped onto real tooling rather than left abstract: **Pydantic** for schema validation _after_ generation (the reject and auto-repair arms); **Outlines** and **LMQL** for constrained decoding _during_ generation (the deterministically-represented arm — the output cannot be malformed because the grammar constrains token selection at the point of generation, not after).
- Exit invariant: no malformed model output reaches a side-effecting executor. Restated precisely once tooling is named: constrained decoding (Outlines/LMQL) satisfies the invariant structurally; schema validation (Pydantic) satisfies it by rejection — only the structural route gets the flat-latency property M12/E28 later formalizes.

---

## Module 11 — Hallucination Protection

### E25 — Hallucination Is a Measurement Problem Before It Is a Model Problem

**Principles Applied:**

- **Signal Detection Theory (classical psychophysics, applied to ML):** every hallucination detector is a threshold-selection problem trading false positives against false negatives, with asymmetric costs depending on the domain.
- **Information Theory / Entropy:** semantic entropy across multiple sampled generations of the same prompt is a genuine uncertainty signal, cheap to compute, cited here as the second-cheapest tier of the detection ladder.
- **Statistical Decision Theory (general):** the formal basis for choosing a detection threshold given known error costs, rather than picking one by feel.

**Key Concepts:** Confidence ≠ correctness · Uncertainty calibration · Claim-level verification.

**Key Points:**

- Detection ladder: cheap confidence signal → consistency/entropy → NLI entailment → calibrated LLM judge → human adjudication — the NOLLM cascade from M3/E10 applied to its flagship use case.
- Live demo: one generated answer is broken into atomic claims, each scored individually against grounding evidence.
- Pays off M5/E16's false-recall preview directly: false recall is shown as this episode's false-positive case, structurally distinct from ordinary forgetting.

### E26 — Grounding Failure vs. Reasoning Failure

**Principles Applied:**

- **Differential Diagnosis (medicine):** correctly separating two conditions with superficially similar symptoms but different causes and different treatments — applied directly to two hallucination failure modes that look identical from the outside.
- **Fault Isolation (general engineering troubleshooting discipline):** the systematic method of narrowing down which subsystem actually caused an observed failure before attempting a fix.

**Key Concepts:** Grounding failure (evidence was never available — a retrieval problem, M5's job) vs. reasoning failure (evidence was available but misused — E25's detection ladder's job).

**Key Points:**

- Debug path taught explicitly: did retrieval contain the necessary evidence? If yes, it's a reasoning problem; if no, it's a retrieval problem.
- Explicit warning against the common misdiagnosis: "fixing RAG" (M5/E19) when the real defect is synthesis, or vice versa.

### E27 — Trust Calibration: Automation Bias, Human Review & Hallucination SLOs

**Principles Applied:**

- **Google SRE Error Budgets (Beyer et al., 2016):** hallucination rate is reframed as a tracked, spendable error budget rather than a one-time launch benchmark.
- **Automation Bias (Parasuraman & Riley, 1997, human factors):** humans systematically over-trust systems that appear highly capable, and stop checking them — meaning a human-in-the-loop step can _reduce_ safety if it degrades into reflexive approval-clicking.
- **Algorithm Aversion (Dietvorst et al., 2015):** the opposite failure — humans over-correcting and distrusting a system even when it's statistically more reliable than they are, after having seen it err even once.

**Key Concepts:** SLO · Error budget · Escalation rate · Reviewer calibration.

**Key Points:**

- Key insight stated directly: a human-approval step is not automatically a safety mechanism — if reviewers approve everything, the architecture has produced theatre, not safety.
- Exit metrics: hallucination rate, calibrated confidence, human override rate, time-to-review, disagreement rate — reused directly in M16/E38's Little's Law queue-capacity math.

---

## Module 12 — Guardrails

### E28 — Hard Guarantees vs. Probabilistic Corrections

**Principles Applied:**

- **Formal Language Theory (Chomsky, 1956), reused from M10/E24:** constrained decoding as a grammar enforced during generation — a structural guarantee, not a retry loop.
- **Formal Verification (general discipline):** the sharp distinction between proving a property must hold and merely testing that it happened to hold this time.

**Key Concepts:** Generate → validate → retry (probabilistic, check-then-retry) vs. generate-under-constraint (a hard guarantee) · Structural guarantee ≠ semantic guarantee — a schema can guarantee `amount_usd` is numeric; it cannot guarantee $400 is authorized (that's M4/E12's job, not this episode's).

**Key Points:**

- The module's organizing distinction, stated precisely enough that later episodes don't need to re-derive it.
- Live side-by-side demo comparing retry-loop tail latency against constrained decoding's flat, guaranteed latency — the formal payoff of M10/E24's tooling claim.

### E29 — Guardrails as a System: Chaos, Property Tests & Adversarial Evaluation

**Principles Applied:**

- **Chaos Engineering (Basiri et al., Netflix, 2016):** deliberately injecting adversarial or malformed input in staging, not waiting for production to surface it.
- **Property-Based Testing (Claessen & Hughes, QuickCheck, 2000):** testing that an invariant holds across a generated space of inputs, not a fixed set of hand-picked examples.
- **The Swiss Cheese Model of defense (Reason, 1990, safety engineering):** each individual guardrail layer has holes; the layers are only safe in combination, when one layer's gaps don't align with another's.

**Key Concepts:** Invariant testing · Mutation testing · Attack-path testing · Coverage gap.

**Key Points:**

- Core invariant taught as a diagram: unsafe input may bypass one layer, but must encounter another, and must never reach an unauthorized side effect — the Swiss Cheese Model made concrete for this system.
- Grades guardrail _effectiveness_ using M3's NOLLM scoring stack, not merely guardrail _presence_.

---

## Module 13 — Security: Prompt Injection, PII, OWASP

### E30 — Prompt Injection Is a Control-Flow Problem

**Principles Applied:**

- **Defense in Depth (military/security doctrine):** no single layer is assumed sufficient; independent layers each catch what the others miss.
- **Zero Trust, reused from M4/E12:** untrusted input is never granted trust based on its apparent source or plausibility.
- **Complete Mediation, reused from M4/E12:** every injected instruction still has to pass through the same Gatekeeper as any other proposal — it gets no shortcut.
- **STPA, previewed:** briefly framed here as "an unsafe outcome can emerge from individually correct components interacting" — fully developed in a future safety-engineering treatment.

**Key Concepts:** Instruction/data separation · Tool scoping.

**Key Points:**

- Central reframe: prompt injection is not solved by "better prompts" — it's a hostile attempt to alter control flow, and the fix is architectural, not linguistic.
- Strongest control stated as the episode's thesis: if injected text cannot reach an authority-bearing execution path, successful _detection_ is not even required for safety — the architectural difference between detection and prevention, and the same idea M4/E12 already built the Gatekeeper to guarantee.

### E31 — OWASP Agentic Risks as a Control Mapping Exercise

**Principles Applied:**

- **OWASP Top 10 for Agentic Applications (2026):** the current, peer-reviewed industry framework, treated here as an input to a control-mapping exercise rather than a lecture to be read aloud.
- **FMEA, previewed:** used here for severity ranking of each mapped risk.
- **Least Privilege, reused from M4/E12:** OWASP's "Excessive Agency" — its own top-ranked agentic risk — is mapped directly and explicitly onto a Principle of Least Privilege violation, not treated as a novel LLM-era problem requiring novel theory.

**Key Points:**

- For every OWASP risk: threat → failure mode → existing control (already built in an earlier episode) → coverage gap → eval → residual risk.
- The audience leaves with a control matrix, not ten memorized definitions — and an honest accounting of which risks are genuinely covered by prior modules' work versus genuinely still open (Sensitive Information Disclosure is flagged here as the open item E32 closes next).

### E32 — PII & Sensitive Data: Detection, Redaction, Minimization **[NEW]**

**Principles Applied:**

- **Privacy by Design (Cavoukian, formalized 1990s, seven foundational principles):** privacy is an architectural property designed in from the start, not a cleanup phase applied after the fact.
- **Data Loss Prevention pattern taxonomy (general information-security practice):** the standard three-part discipline — detection, redaction, minimization.
- **Least Privilege, reused a second time, now applied at the data layer instead of the action layer:** the same principle E12 applied to _what an action can do_ is applied here to _what a tool call or a log line is allowed to see_.

**Key Concepts:** Detection (pattern/NER-based PII classifiers) · Redaction (masking vs. tokenization vs. removal, and their different reversibility guarantees) · Minimization (collect/expose/retain the least necessary).

**Key Points:**

- Explicitly distinguishes three PII surfaces that get different treatment: PII-in-input (what a tool call is allowed to see, governed by this episode's minimization rule), PII-in-memory (M5/E16's provenance tagging, reused directly), PII-in-logs (forward reference to M16/E36, where redaction-before-persistence is enforced at the tracing layer).
- Directly closes the coverage gap E31 left open: OWASP's Sensitive Information Disclosure risk is mapped to this episode's controls, completing E31's honest accounting.
- Exit artifact: a `PIIFilter` component with detection, redaction, and minimization stages, wired into both the tool-execution path and (forward reference) the tracing path.

### E33 — Data Lineage & Regulator-Ready Evidence Chains

**Principles Applied:**

- **Data Governance and Lineage discipline (general regulatory-compliance practice):** the source discipline for tracking what every consequential decision was based on.

**Key Points:**

- Data minimization at every layer, reused from E32: what's collected, what's exposed to each tool, what's retained, what's copied between contexts, what's accessible to agents without genuine need.
- Architecture rule stated directly: every consequential decision should be attributable to a versioned set of model, policy, data, and authority conditions — a regulator-ready evidence chain, not an afterthought. This is the artifact M16/E36's tracing schema is required to actually produce, and the evidence M16/E37 later uses to score the harness's own CMMI-style maturity honestly rather than asserting it.

---

## Module 14 — Sandbox Execution

### E34 — Secure Execution: Isolation Strength, MicroVMs & the Blast-Radius Budget

**Principles Applied:**

- **Popek & Goldberg's Formal Virtualization Requirements (1974):** the classical theoretical basis for what actually constitutes genuine isolation versus merely apparent isolation.
- **Least Privilege and Defense in Depth, reused:** isolation strength is one more layer in the same stack, not a separate discipline.

**Key Concepts:** Isolation strength / startup-latency tradeoff.

**Key Points:**

- Tiers: process policy → OS sandbox → container → microVM → remote isolated worker.
- Reframes the design question from "which sandbox is best" to "what isolation strength is justified by this workload's blast radius (M0/E03) and threat model," measured against real workload data rather than architectural fashion.

---

## Module 15 — Resilience: Systems Engineering

### E35 — Resilience Is a Control Plane Too: Timeouts, Retries, Circuit Breakers, Bulkheads & Sagas

**Principles Applied:**

- **Distributed Systems Theory (general):** once an agent calls multiple services, retries, waits, persists, and resumes, it has inherited every distributed-systems failure mode whether or not it was built to expect them.
- **Nygard's _Release It!_ (2007):** the source of the Circuit Breaker and Bulkhead patterns as production-hardening disciplines, not academic curiosities.
- **The End-to-End Principle (Saltzer, Reed & Clark, 1984):** some guarantees are only meaningful at the true endpoint of a system, not at any intermediate hop — a transport layer confirming a message arrived is not the same as the application confirming the refund was committed exactly once, for the right order, under the right authority.

**Key Concepts:** Timeout · Retry (only safe when paired with idempotency) · Backoff · Idempotency key · Circuit Breaker (Closed → Open → Half-Open as a small finite state machine — the canonical worked example of M9/E23's state-table discipline applied to a dependency) · Bulkhead (isolated resource pools) · Saga (a chain of local transactions with defined compensating actions).

**Key Points:**

- Central teaching point: retry is not a recovery strategy, it's a resource-consuming hypothesis that a failure is transient — and it's only safe when the retried action is idempotent.
- Saga demo on the Refund Agent: validate → reserve amount → issue refund → update order → notify customer, with a simulated crash injected after step 2 or 3, and a live compensating rollback.
- Bulkhead demo: one failing external provider is shown unable to exhaust the capacity budget of an unrelated dependency.
- Exit invariants: retries are idempotent; repeated failure eventually stops the call; one dependency cannot exhaust another's budget; every partial transaction has a defined compensation path.

---

## Module 16 — Observability, Logging, Monitoring & Operations

### E36 — Observability for Agents: Trace the Decision, Not Just the Request

**Principles Applied:**

- **Distributed Tracing / Dapper (Sigelman et al., Google, 2010):** the architectural basis for propagating a single trace ID through every hop of a distributed request.
- **Chain of Custody (forensic and legal practice):** an audit trail must be able to reconstruct not just that something happened, but the unbroken sequence of custody/decisions that led to it — the standard this episode holds agent traces to.

**Key Concepts:** Every hop gets a trace ID, actor, authority context, tool, policy version, model version, state transition, result, latency, and cost.

**Key Points:**

- Key principle stated directly: an audit log that cannot reconstruct _why_ a side effect happened is not an audit trail — it's telemetry, and the two are not interchangeable for a regulator or a CISO.
- Explicit requirement, stated on camera: traces must pass through E32's `PIIFilter` before persistence — a direct cross-module dependency, not left implicit.

### E37 — SRE for Agents: SLOs, Error Budgets & Incident Response

**Principles Applied:**

- **Google Site Reliability Engineering (Beyer et al., 2016):** SLOs and error budgets as the operating discipline, extended from uptime (the classical target) to agent-specific targets.

**Key Concepts:** Agent SLOs — task success, unauthorized-action rate, hallucination rate, escalation rate, loop-abort rate, tool failure rate, cost per successful task, recovery success, policy violation rate.

**Key Points:**

- Postmortem format taught as a fixed template: impact → trigger → missing control → detection gap → containment → root cause → permanent eval.
- Central discipline: every real incident becomes a future regression test, closing the loop back to M3.
- **Closes M0/E04's CMMI thread:** the episode ends by scoring the series' own reference harness against E04's maturity ladder using only the evidence this module's SLOs, postmortems, and E33's audit trail actually produce — proving the "don't self-declare a level" promise from E04 with real telemetry, not an assertion.

### E38 — Human Review Is a Queue: Little's Law

**Principles Applied:**

- **Little's Law (Little, 1961, queueing theory):** for a stable queue, `L = λW` — the number of items waiting equals the arrival rate times the average wait time, an exact relationship, not a rule of thumb.

**Key Points:**

- Worked example: a system generating 100 approval cases/hour, each taking 2 minutes to review, already requires more than 3.3 reviewer-hours per hour of incoming work — a capacity requirement most teams never compute before shipping a human-in-the-loop design.
- Key insight stated directly: human-in-the-loop is an operational staffing system, not a UI button, and M0/E03's automation-level decisions directly determine the arrival rate this queue has to absorb.
- Reuses M11/E27's exit metrics (escalation rate, time-to-review) as this episode's queue-arrival and service-time inputs.

### E39 — Shipping Agent Changes Safely: Version Everything That Can Change Behavior

**Principles Applied:**

- **The Twelve-Factor App (Wiggins, Heroku, 2011):** the source discipline for externalized configuration and stateless processes, applied here to agent-specific artifacts.
- **Progressive Delivery / Canary Release (modern continuous-delivery practice):** gradual, measured traffic shift to a new version rather than a hard cutover.

**Key Points:**

- Versioned separately: prompt (a DSPy-compiled artifact per M7/E21), tool schema, policy, evaluator, model, retrieval index, memory schema, guardrail, and PII-redaction rules from E32.
- Rule stated directly: never a hard cutover on the Gatekeeper's live authority path; rollback must be possible at the same granularity as deployment.

---

## Module 17 — Multi-Agent Architecture

### E40 — Do You Even Need Multi-Agent?

**Principles Applied:**

- **Reliability Block Diagrams / Series-Parallel Reliability Math (classical reliability engineering):** for a series chain, `P(success) = Π P(component_i succeeds)` — a single 90%-reliable agent chained four times in series lands at ≈65.6%, an immediate, sobering number that should change how "more agents = more intelligence" gets talked about.
- **Fault Tree Analysis, previewed:** briefly introduced here as the tool for reasoning backward from a catastrophic outcome to combinations of contributing failures.

**Key Concepts:** Compounding accuracy · Series chain vs. parallel/redundant composition.

**Key Points:**

- Worked cost comparison: an 8× cost increase for zero net benefit in a realistic series-chain scenario.
- Legitimate reasons for multiple agents taught explicitly — independent verification, specialization, context isolation (M5), parallel search, failure containment — contrasted against the illegitimate reason ("it sounds more autonomous").
- This episode is the economics payoff for M1/E07's orchestrator-subagent pattern-shape preview.

### E41 — Multi-Agent Topologies: Supervisor, Hierarchy, Peer Network, Swarm

**Principles Applied:**

- **Organizational Design Theory (Mintzberg, 1979):** Supervisor vs. Swarm topology maps directly onto centralized-hierarchy vs. distributed-network organizational structures — a discipline with decades of empirical grounding on when each structure actually works.
- **Conway's Law (1967):** system structure tends to mirror organizational communication structure; a multi-agent topology designed purely on a whiteboard, without asking who owns which domain, tool, or policy, will fight the organization that has to run it.

**Key Concepts:** Centralized supervisor · Hierarchical delegation · Peer network · Event-driven workers.

**Key Points:**

- Decision variables taught as a checklist: coordination cost, authority boundaries, shared state, conflict resolution, observability, failure propagation.
- Explicit exercise mapping organizational ownership (who owns the domain, the tools, the policy, conflict resolution) onto the technical topology, per Conway's Law.

### E42 — Agent Communication: MCP, A2A & Event-Driven Architecture

**Principles Applied:**

- **Protocol Layering (the OSI-model lineage):** the formal separation between transport-level delivery and application-level semantics, applied to the distinction between MCP (agent-to-tool) and A2A (agent-to-agent).
- **Lamport's Happened-Before Relation, reused from M9/E23:** causal ordering across concurrent agent-to-agent messages, not just timestamps.

**Key Concepts:** MCP (agent-to-tool protocol) · A2A (agent-to-agent protocol) · Correlation ID · Idempotent delivery semantics.

**Key Points:**

- Key principle stated directly: do not confuse "agent communication" with "agents passing prose to one another" — a production protocol defines identity, schema, authority, correlation, retry semantics, and failure semantics.
- Live demo standing up a minimal MCP server, and the case for event-driven (Kafka-style) coordination over direct point-to-point calls once the system scales past a handful of agents.

### E43 — Designing for Removal: Multi-Agent Complexity Is Debt

**Principles Applied:**

- **The Bitter Lesson (Sutton, 2019):** general methods that scale with compute reliably outlast hand-engineered scaffolding built to compensate for a weaker model — scaffolding that compensated for a weaker underlying model becomes dead weight as the model improves.
- **Brooks' Law and Coordination Cost (Brooks, _The Mythical Man-Month_, 1975):** adding people — or agents — to a system doesn't automatically add throughput; it adds communication, supervision, synchronization, and duplicated work.

**Key Concepts:** Coordination cost · Architectural debt.

**Key Points:**

- Live refactor: starts with router agent / policy agent / refund agent / review agent, removes two, and proves the remaining system preserves authority boundaries, tests, auditability, and cost model.
- Principle stated directly: every additional agent must earn its existence with a measurable architectural benefit, per E40's math — this episode is that principle's live demonstration.

---

## Module 18 — Economics: Build vs. Buy

### E44 — Cost & Performance Engineering: Amdahl, Caching, Cascades & Budgets

**Principles Applied:**

- **Amdahl's Law (Amdahl, 1967):** a small amount of non-parallelizable, non-scalable work — human approvals, serial tool calls, single-threaded orchestration — can dominate overall system performance regardless of how much the rest of the system is optimized.
- **Cascade Classifiers, reused from M3/E10:** model cascading (cheapest sufficient tier first) applied directly to cost, not just to eval scoring.
- **Cache Economics / Locality of Reference, reused from M5/E18.**

**Key Points:**

- Cost stack made explicit: input tokens + output tokens + tool calls + retrieval + evaluator calls + human review + orchestration + infrastructure — most teams optimize only the first two and miss the majority of real cost.
- Key principle stated directly: optimize cost per successful business outcome, not dollars per model call — a cheaper-but-wronger call is not a saving.

### E45 — Build vs. Buy: Choose the Control Surface, Not the Vendor

**Principles Applied:**

- **Transaction Cost Economics (Coase, 1937; Williamson, 1975/1985):** the classical make-or-buy framework — a decision should weigh the cost of coordinating through the market (buying) against the cost of coordinating internally (building), not brand reputation or feature checklists.

**Key Points:**

- Evaluation dimensions: control over execution, state semantics, isolation, observability, extensibility, policy enforcement, deployment model, portability, data boundary, recovery semantics.
- Central reframe: the question is not "which vendor" but "which control surfaces must we own" — framework/platform survey content kept deliberately short and refreshable, since it's the fastest-decaying content in the series.

### E46 — The Economics of Autonomy: ROI, TCO & the Cancellation Test

**Principles Applied:**

- **Total Cost of Ownership methodology (Gartner, 1987):** the discipline for accounting the full cost of a system beyond its sticker price.
- **Risk-Adjusted Economics (general decision theory):** expected value calculated under genuine uncertainty, not a point estimate that ignores residual risk.

**Key Points:**

- Full cost accounting: business value − model cost − retrieval cost − tool/infrastructure cost − evaluation cost − human review cost − incident cost − integration cost, then risk-adjusted.
- Final metric handed to the audience as the number executives can actually use: cost per successful, policy-compliant, recoverable business outcome.

---

## Module 19 — Miscellaneous

### E47 — Loose Ends: What Doesn't Fit a Single Pillar

A short-format episode (not full 30–60 min if the content doesn't warrant it) collecting genuinely cross-cutting items that resist a single-pillar home, reviewed and updated as later modules are actually scripted.

**Key Points:**

- Any series-wide continuity items still open at script time (naming collisions between two episodes' versions of the same mechanism, forward references that were never paid off) get resolved and documented here rather than silently left inconsistent.
- Candidate future content: anything from OWASP's 2026 list, DSPy's ecosystem, or the vector-DB rubric (M5/E19) that goes stale between recording and release gets its correction filed here rather than requiring a re-shoot of the original episode.
- If, once scripting begins, nothing genuinely orphaned turns up, this module can be cut and the capstone renumbers down by one.

---

## Module 20 — Capstone: The Architecture Defense (three gates, no new principles)

_(No new principles are introduced in this module, deliberately — every one of the preceding grounding principles is expected to be recognized, named, and applied live. That absence of new theory is itself the point: it proves the architecture composes.)_

### E48a — Capstone Gate A: Architecture Review

**Key Points:**

- Designs the Refund Agent from first principles end to end: user → intent → agent reasoning → context → tools → sandbox → Gatekeeper → execution → state → audit.
- Must explicitly defend, on camera, in a design-review format: automation level per action (M0/E03), authority model (M0/E02, M4/E12), role model (M4/E12), tool boundaries (M8/E22), memory model (M5), eval strategy (M3), failure handling (M9/E23, M6/E20), human review design (M11/E27, M16/E38).

### E48b — Capstone Gate B: Adversarial Production Readiness Review

**Key Points:**

- Full attack suite run against the completed system: prompt injection (M13/E30), authority escalation (M0/E02), structuring (M0/E02), replay, duplicate tool calls, poisoned memory (M5/E16), stale policy, compromised tool, malformed output (M10/E24), induced loop (M6/E20), simulated crash (M9/E23), concurrent action, reviewer overload (M16/E38).
- Explicit reframe of the goal: not "nothing ever fails," but "when the system fails, the failure is bounded, detectable, recoverable, and non-authorizing" — the entire series compressed into one sentence.

### E48c — Capstone Gate C: Day-2 Operations & Productization

**Key Points:**

- Shows the transition from demo → production workload → multiple tenants → policy configuration → versioned controls (M16/E39) → fleet observability (M16/E36) → incident response (M16/E37) → economics (M18).
- Proves the architecture survives a model upgrade, a tool upgrade, a policy change, a context-window change, a memory schema change, increased traffic, and tenant-specific permissions — closing the series exactly where M0/E04's maturity ladder and M18/E46's economics opened it: autonomy is only worth what it can be shown, under attack and under load, to actually cost and actually hold.

---

## Full Episode Count & Module Map

| #   | Module                       | Episodes | Count       |
| --- | ---------------------------- | -------- | ----------- |
| 0   | Foundations                  | E01–E04  | 4           |
| 1   | Agentic Design Patterns      | E05–E07  | 3           |
| 2   | Harness Engineering          | E08      | 1           |
| 3   | Evals                        | E09–E11  | 3           |
| 4   | Gatekeeper Architecture      | E12      | 1           |
| 5   | Memory                       | E13–E19  | 7           |
| 6   | Loop Engineering & Detection | E20      | 1           |
| 7   | Prompt Engineering / DSPy    | E21      | 1           |
| 8   | Tool Calling & Registry      | E22      | 1           |
| 9   | State Management / FSM       | E23      | 1           |
| 10  | Output Parsing               | E24      | 1           |
| 11  | Hallucination Protection     | E25–E27  | 3           |
| 12  | Guardrails                   | E28–E29  | 2           |
| 13  | Security                     | E30–E33  | 4           |
| 14  | Sandbox Execution            | E34      | 1           |
| 15  | Resilience                   | E35      | 1           |
| 16  | Observability & Operations   | E36–E39  | 4           |
| 17  | Multi-Agent Architecture     | E40–E43  | 4           |
| 18  | Economics: Build vs. Buy     | E44–E46  | 3           |
| 19  | Miscellaneous                | E47      | 1           |
| 20  | Capstone                     | E48a–c   | 1 (3 gates) |
|     | **Total**                    |          | **48**      |
