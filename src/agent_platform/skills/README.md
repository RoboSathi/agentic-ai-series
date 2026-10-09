# skills — the profile and procedures in an agent's prompt (ADR 0010)

An agent's `AGENT.md` is its **profile**: what it is for, its scope, and how it works, uses
tools, treats memory and escalates. The profile is always in the system prompt.

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
procedural = ProceduralStore(platform) if features.is_on("memory.procedural") else None
instructions = build_agent_instructions(
    settings.skills, store=procedural, audit=ledger, reporter=reporter
)
system = instructions.compose(signals, query).text  # profile + this turn's procedures
```

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
