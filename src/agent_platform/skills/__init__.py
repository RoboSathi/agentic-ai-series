"""The profile and procedures in an agent's prompt (ADR 0010, LLD §8).

An agent's `AGENT.md` is its profile, always in the system prompt. Its
`skills/<name>/SKILL.md` files are procedures: synced into procedural
memory at start-up, selected per turn by the harness, and audited by the
version each turn used.
"""

from agent_platform.skills.catalog import SkillCatalog
from agent_platform.skills.document import (
    SkillDocumentError,
    load_profile,
    load_skill_documents,
    parse_skill,
    parse_stored,
)
from agent_platform.skills.instructions import (
    AgentInstructions,
    InstructionBuilder,
    build_agent_instructions,
    render_profile,
)
from agent_platform.skills.selector import SkillSelector
from agent_platform.skills.types import (
    ActivatedSkill,
    Activation,
    AgentProfile,
    Instructions,
    Skill,
    SkillDocument,
    SkillSource,
    SkillSyncReport,
)

__all__ = [
    "ActivatedSkill",
    "Activation",
    "AgentInstructions",
    "AgentProfile",
    "InstructionBuilder",
    "Instructions",
    "Skill",
    "SkillCatalog",
    "SkillDocument",
    "SkillDocumentError",
    "SkillSelector",
    "SkillSource",
    "SkillSyncReport",
    "build_agent_instructions",
    "load_profile",
    "load_skill_documents",
    "parse_skill",
    "parse_stored",
    "render_profile",
]
