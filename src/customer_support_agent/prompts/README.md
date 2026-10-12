# prompts — what the model is told

`AGENT.md` (the profile), `skills/*/SKILL.md` (one per task, with prefetch reads and tools), the reply prompts, and `customer_policy.md`. The prompt is not a control — the Gatekeeper is.

- **Entry points:** `AGENT.md`, `skills/`, `build_system_prompt`, `build_reply_system_prompt`, `policy_text`
- **Configuration:** `skills.profile_path`, `skills.skills_dir`, `customer_support_agent.policy_path`
- **Design:** [ADR P-12](../../../design_docs/ADR.md#p-12--the-agent-profile-and-skills-are-procedural-memory)
