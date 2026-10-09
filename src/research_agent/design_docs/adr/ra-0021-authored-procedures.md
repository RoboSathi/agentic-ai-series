# ADR ra-0021 — Authored procedures: the profile and role skills

**Status:** accepted 2026-10-09 · amends the "procedural is disabled by design" posture · applies platform ADR 0010

## Context

This agent declared `memory.procedural: false` because it learns nothing procedural across
runs: it has no dream, and no curator queue. Platform ADR 0010 separates two things that were
conflated there. One is *learned* procedures (dream lessons). The other is *authored*
procedures (SKILL.md files in git, synced into procedural memory and versioned there).

## Decision

- **`memory.procedural: true`, for authored procedures only.** The builder builds a
  `ProceduralStore` and syncs `prompts/skills/` into it. Nothing writes a learned procedure,
  because there is no dream.
- **The profile (`prompts/AGENT.md`) goes directly below the `ROLE:` line** for the roles in
  `research_agent.instructions.profile_roles`: the interviewer, planner, researcher, reflector,
  synthesizer and answerer. Each such call also carries the procedure its `role:{role}` signal
  selects (`research-planning`, `source-verification`, `report-synthesis`). `role_of` still
  reads the role from the first line.
- **The judge and the extractor are left narrow.**
  - The judge audits the run, and prompting it with the agent's own profile would weaken that
    independence (ra-0003).
  - The extractor runs once per page over untrusted text. It needs its narrow quarantine
    instructions, and the profile would multiply its token cost.
- Each role's own instructions stay authoritative for its output schema. A procedure only
  adds how the step should be approached.

## Consequences

- Each profile-role call records an activation, de-duplicated per (trace, signals, query).
- Withholding `memory.procedural` returns the roles to their bare instructions plus the
  profile.
