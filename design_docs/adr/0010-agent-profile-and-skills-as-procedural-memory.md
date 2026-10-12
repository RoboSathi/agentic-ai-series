# ADR 0010 — The profile and skills as procedural memory

- Status: accepted; implemented 2026-10-09
- Date: 2026-10-09
- Related: ra-0021 (the research agent's authored procedures)

## Context

Each agent's prompt was a hard-coded string. Customer Support's `_INSTRUCTIONS`, the ops agent's
`system_prompt:` in config and the research agent's role constants all mixed three things:
what the agent is for, how it should behave, and step-by-step procedures for particular tasks.
Every turn carried every procedure, whether the turn needed it or not.

At the same time, procedural memory existed only on paper:

- `ProceduralStore` was built by no agent.
- `MemoryFacade.procedure()` had no callers.
- The `memory.procedural` toggle was read nowhere.
- A procedure the dream proposed could be queued but never approved. `approve_queued` also
  looked up the *semantic* store for every category, so approving a procedure retired any
  semantic fact that shared its key.

## Decision

### Vocabulary

| Term | Meaning |
|---|---|
| **Agent profile** | An agent's `AGENT.md`. It is always in the system prompt, never gated, and versioned by content digest (`profile@<digest>`). It is never the authority for limits, permissions or record state. |
| **Skill** | A procedure authored as a file: `skills/<name>/SKILL.md`, with frontmatter (`name`, `description`, `triggers`, `keywords`), a body (`Preconditions`, `Procedure`, `Boundaries`, `Completion`) and optional `references/*.md` inlined into it. |
| **Procedure** | A version in `ProceduralStore`, with one active version per name. Its source is `authored` (a synced skill) or `consolidated` (a dream lesson a curator approved). |
| **Signal** | An opaque string an agent derives from its turn, such as `intent:REQUEST_REFUND`, `route:llm_fallback` or `role:planner`. The platform matches signals and never interprets them. |
| **Trigger** | A signal a skill declares. The reserved trigger `always` (named in config) is active on every turn. |
| **Activation** | Which procedures one turn's prompt carried, as `name@version`, each with a reason (`always`, `trigger:<signal>` or `lexical:<score>`). |

### Mechanism (`agent_platform/skills/`, beside `context/`)

1. **Sync at build time.** The composition root calls `build_agent_instructions(settings.skills,
   store=…, audit=…, reporter=…)`.
   - The profile is loaded and its sections are checked.
   - When the agent's posture includes `memory.procedural`, every SKILL.md is parsed and
     validated, then written into `ProceduralStore`. A new version is written only when the text
     differs from the *latest* version, so an operator's rollback survives a restart.
   - Authored skills whose files are gone are retired. Consolidated lessons are never touched.
   - Sync is a deployment step, not a principal's claim, so it writes directly rather than
     through the governed write pipeline. Each change is recorded as a `procedure_sync` audit
     entry.
   - A malformed file fails the build.
2. **Select deterministically, with no model call.** A turn's procedures are chosen in this
   order:
   - every `always` skill;
   - every skill with a trigger in the turn's signals;
   - only if neither of those produced a non-`always` match, BM25 (`context.retrieval.LexicalRetriever`)
     over description and keywords, above `skills.selection.min_lexical_score`.

   The result is capped at `skills.selection.max_active`. Consolidated lessons have no triggers,
   so only the lexical step can select them.
3. **Place procedures outside the context window.** `AgentInstructions.compose(signals, query)`
   renders the profile, then `# Procedure: <name>` sections, into the system message. They are
   not a `ChunkKind`: they are never evicted by the token budget, they still arrive with
   `context` off, and they are not screened as data. Skill text is never grounding evidence (a
   procedure is not a fact). The profile stays as `system` evidence.
4. **Record once per turn.** Each turn emits one `skill_activation` card (PROCEDURES) and one
   `procedure_activation` audit entry under the turn's trace, holding the profile ref, the
   procedure refs and their reasons. This needs no audit schema migration.
5. **Gate.** The profile is always present. Procedures are gated by the existing
   `memory.procedural` toggle; there is no tenth feature. Withheld, nothing is synced and the
   prompt carries the profile alone.
6. **Approve lessons with `/skills`.** The platform `/skills` command offers `list`, `show`,
   `history` and `approve <queue id>`. Approval goes through the governed
   `MemoryWritePipelineImpl.approve_queued`, which now supersedes through the semantic store only
   for semantic categories. Customer Support builds `ProceduralStore` (for the agent and for the
   dream), so an approved lesson becomes an active, consolidated procedure.

### Per agent

| Agent | Agent profile | Skills | Signals |
|---|---|---|---|
| Customer Support | `prompts/AGENT.md` | `order-status`, `order-details`, `list-orders`, `refund-workflow`, `refund-status`, `cancel-order` | `intent:{intent}`, `route:{intent_execution_route}`, `intent:{intent_suggested}` for an unrouted turn's read-intent hint (`customer_support_agent.instruction_signals`) |
| Ops | `prompts/AGENT.md` | `analytics-investigation` (`always`) | — |
| Research | `prompts/AGENT.md`, below each profile role's `ROLE:` line | `research-planning`, `source-verification`, `report-synthesis` | `role:{role}` for roles in `research_agent.instructions.profile_roles` |

## Consequences

- The refund rules now arrive with the refund procedure rather than on every turn.
  - An unrouted turn (`route:llm_fallback`) loads **no skill by default** (amended 2026-10-12).
    It gets one only when its words name a task (lexical, over content words: function words
    are dropped first, `skills.selection.function_words`) or when it nearly reached a *read*
    intent (`intent_router.suggestion`: `floor`, `margin`). A skill that changes something is
    never loaded on an unrouted turn, and the turn is offered `intent_router.unknown_tools`
    at most: no skill must never mean every tool.
  - Keywords mirror the intent phrases, for when classification is withheld.
  - Reply-phrasing rules ("issued, settles shortly") stay in the profile, so grounding evidence
    is unchanged.
  - `instruction_corpus()` (profile, all procedures, policy, reply prompts) is what the leak
    check and the refund-limit test read.
- A turn resumed after a restart with edited files re-activates the newer versions. Both
  activations are on the audit trail.
- A legacy `compiled_prompt` row left in a database is never offered:
  `skills.selectable_artifact_types` is an allow-list.
- `/harness` rebuilds re-sync. Sync is idempotent and its card is quiet.

## Not done

- The turn-summary tally and `/versions` do not list procedures yet. The PROCEDURES card and the
  audit entry carry them.
- There is no end-to-end test of dream → `/skills approve` → lexical selection. Each step is unit
  tested.

## Amendment 2026-10-10 — procedures that make agents faster

The first build added procedures *on top of* the old instructions. Every system prompt
grew by 30–70%, and nothing a procedure said saved the model any work. A CSA refund turn
still took 4 model calls (read the order, read the refund, request it, reply), at 52–90 s.

- **"Charter" is now "agent profile"** (`AgentProfile`, `skills.profile_path`,
  `profile@<digest>`). The profile is identity and non-negotiables only. Its required
  sections are Mission, Scope and Rules.
- **Lean prompts.** Only `skills.render.sections_sent` (Procedure, Boundaries) reaches the
  model; Preconditions and Completion are kept for maintainers.
  - `skills.max_prompt_chars` per agent fails the build if the profile plus the largest
    procedures outgrow it: CSA 2,400 (its old instructions were 3,167), ops 1,500 (old 1,493).
  - Measured worst cases are now CSA 2,186 and ops 1,206.
- **Prefetch.** A skill declares `prefetch:` reads. The harness runs them before the first
  model call, through the governed tool path, and hands the model their results as its own
  calls. The tool request cards say "Prefetched: … · for <procedure>".
  - CSA has a `prefetch` graph node before `propose`. Arguments come from turn state
    (`customer_support_agent.prefetch.arguments`). A read whose arguments are not known is
    skipped.
  - Ops prefetches `list_metrics` and `list_views`.
  - A refund turn is now 2 model calls.
- **Tools.** A skill declares the `tools:` it uses. On an intent-routed turn the model is
  offered only those, intersected with the intent's tools; an unrouted turn is narrowed by
  its procedures too (order-status only, so reads). A read prefetch already answered is
  not offered again, nor is `list_orders` once the order is known.
  - Every tool has a *model name* (`tools.model_names`, else its bare name). The model and
    the cards never see `namespace__name`.
  - Qualified names stay in policy, audit, facts and evals; `qualified_tool` is resolved
    back from the model name.
  - The build refuses a procedure that names a tool the agent does not offer, prefetches one
    with an effect, or writes an internal `namespace.name`.
- **The PROCEDURE card** is drawn only when a procedure is used, once per turn. Its heading:
  `refund-workflow v3 · chosen for intent REQUEST_REFUND · harness reads first: get_order_status,
  get_refund_status · model may call: process_refund`; its body lists each procedure's steps as
  written, then `harness_reads_first` and `model_may_call`. Nothing is drawn when none matched. A sync card appears only when
  something changed.
- **Research agent**: its role skills duplicated each role's own instructions, so they are
  removed (ra-0021, amended). The profile goes on the direct answer only.
