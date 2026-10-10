# ADR ra-0021 — The agent profile on the direct answer; no role procedures

**Status:** accepted 2026-10-09, amended 2026-10-10 · applies platform ADR 0010

## Context

The first build gave this agent three role skills (`research-planning`, `source-verification`,
`report-synthesis`). It also placed the agent profile below the `ROLE:` line of every role
except the judge and the extractor. Each role's own instructions already *are* its procedure,
so both were a second copy. They added about 2.2k characters to every interviewer, planner,
reflector and synthesizer call and saved nothing.

## Decision

- **`memory.procedural: false`.** This agent composes no procedures: it has no dream, and its
  roles' instructions are their procedures.
- **The profile (`prompts/AGENT.md`) goes on the direct answer only**
  (`research_agent.instructions.profile_roles: [answerer]`). That prompt has no role text for
  the profile to duplicate.
- `StructuredCaller` keeps its preamble hook, below the `ROLE:` line so `role_of` still reads
  the role, for any role later given one.

## Consequences

- Run-role prompts are back to their own instructions alone.
- No skills are synced for this agent, and no PROCEDURE card is drawn.
