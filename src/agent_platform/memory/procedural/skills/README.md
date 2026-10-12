# memory/procedural/skills — the agent profile and skills

Parses and validates `AGENT.md` and `skills/<name>/SKILL.md`, syncs skills into `SkillStore` at build time, and selects a turn's skills without a model call (always ∪ triggered, else BM25). Skills replace work: prefetched reads, narrowed tools, lean prompts.

- **Entry points:** `SkillDocument`, `SkillCatalog`, `SkillSelector`, `AgentInstructions`, `build_agent_instructions`
- **Configuration:** `skills:`
- **Design:** [LLD §A16](../../../../../design_docs/LLD.md#a16-memory) · [ADR P-12](../../../../../design_docs/ADR.md#p-12--the-agent-profile-and-skills-are-procedural-memory)
