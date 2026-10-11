# skills — the profile and procedures in an agent's prompt (ADR 0010)

An agent's `AGENT.md` is its **agent profile**: who it is, what it handles, and the rules it
never breaks (sections Mission, Scope, Rules). The profile is always in the system prompt, and
it is kept short: anything task-shaped is a procedure.

Its `skills/<name>/SKILL.md` files are **procedures**:

- They are synced into procedural memory (`memory/procedural/`) at build time.
- One version per name is active.
- Each turn's are selected by the harness, with no model call.
- Each turn's are audited as `name@version`.

```
skills/
  types.py         SkillDocument, Skill, Agent profile, Activation, Instructions, SkillSyncReport
  document.py      parse + validate AGENT.md / SKILL.md (frontmatter via yaml.safe_load)
  catalog.py       SkillCatalog: sync files → store (new version only on change), active, history
  selector.py      SkillSelector: always ∪ triggered, else BM25 over description + keywords, capped
  instructions.py  AgentInstructions (profile_section, compose) and build_agent_instructions
```

## Using it from a composition root

```python
procedural = SkillStore(platform) if features.is_on("memory.procedural") else None
instructions = build_agent_instructions(
    settings.skills, store=procedural, audit=ledger, reporter=reporter
)
activation = instructions.select(signals, query)  # no model call
offered = tools_this_turn_offers(activation)  # narrowed by activation.tools(), really sent
# The card first — the procedures, the reads they do first, the tools really offered —
# so no read ever appears without the procedure that asked for it.
instructions.announce(activation, prefetched=activation.prefetch(), offered=offered)
# … then run the activation.prefetch() reads through the tool path, hand results to the model …
system = instructions.render(activation)  # profile + the procedures' sent sections
```

`compose(signals, query)` does select, announce and render in one step, for an agent with no
prefetch.

## What makes a procedure fast

- **prefetch:** the reads it always starts with. The harness runs them before the first
  model call, so the model skips those rounds.
- **tools:** what it uses. The model is offered only these, by model name.
- Only the Procedure and Boundaries sections are sent (`skills.render.sections_sent`). Keep
  steps short and never repeat the profile.
- `skills.max_prompt_chars` fails the build if the profile plus the largest procedures grow
  past the agent's budget.

The agent overlays `skills.profile_path` and `skills.skills_dir`, and derives its own
**signals**: opaque strings a skill's `triggers` name, such as an intent or a role. The
platform matches signals and never interprets them.

## A skill

```markdown
---
name: example-task            # must equal its folder; skills.name_pattern
description: When this applies and what it achieves.
triggers: [some:signal]       # or [always]
keywords: [words, a, query, uses]
prefetch: [look_up_thing]     # reads run before the model is asked (model names)
tools: [look_up_thing, act]   # what a turn running this is offered
---
## Preconditions
## Procedure
## Boundaries
## Completion
```

Files under `references/*.md` are appended to the stored procedure, so one version covers
both. A malformed file fails the build.

## What it is not

- Not a `ChunkKind`. Procedures sit outside the context window: never evicted, present with
  `context` off, and not screened as data.
- Not grounding evidence. A procedure is not a fact.
- Not an authority. Limits, permissions and record state belong to code, policy and tools.
