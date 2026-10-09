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
| Customer Support | `prompts/AGENT.md` | `order-status`, `order-details`, `list-orders`, `refund-workflow`, `refund-status`, `cancel-order` | `intent:{intent}`, `route:{intent_execution_route}` (`customer_support_agent.instruction_signals`) |
| Ops | `prompts/AGENT.md` | `analytics-investigation` (`always`) | — |
| Research | `prompts/AGENT.md`, below each profile role's `ROLE:` line | `research-planning`, `source-verification`, `report-synthesis` | `role:{role}` for roles in `research_agent.instructions.profile_roles` |

## Consequences

- The refund rules now arrive with the refund procedure rather than on every turn.
  - An unrouted turn (`route:llm_fallback`) still activates `refund-workflow` and `order-status`.
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
