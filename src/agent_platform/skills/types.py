"""The vocabulary of the profile and procedures in a prompt (ADR 0010).

A *skill* is a procedure authored as a file. A *procedure* is a skill (or
a curator-approved lesson) once it is a version in procedural memory. An
*activation* is which procedures one turn's prompt carried, and why.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Where a procedure came from: a file in the repository, or a lesson the
# dream proposed and a curator approved.
SkillSource = Literal["authored", "consolidated"]


class SkillDocument(BaseModel):
    """One procedure's text, parsed: what it is for and what it says."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    triggers: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    # What the model reads: the procedure without its frontmatter.
    body: str
    # What is stored and digested: the file as written, references included.
    raw: str


class Skill(BaseModel):
    """A procedure as procedural memory holds it: a document at a version."""

    model_config = ConfigDict(frozen=True)

    document: SkillDocument
    version: int
    source: SkillSource

    @property
    def name(self) -> str:
        return self.document.name

    def ref(self, template: str) -> str:
        """How a record names this exact version (`refund-workflow@3`)."""
        return template.format(name=self.name, version=self.version)


class AgentProfile(BaseModel):
    """An agent's AGENT.md: always in the prompt, versioned by its digest."""

    model_config = ConfigDict(frozen=True)

    text: str
    digest: str


class ActivatedSkill(BaseModel):
    """A procedure chosen for one turn, and the reason it was."""

    model_config = ConfigDict(frozen=True)

    skill: Skill
    reason: str


class Activation(BaseModel):
    """Everything procedural one prompt carried."""

    model_config = ConfigDict(frozen=True)

    profile: AgentProfile
    skills: tuple[ActivatedSkill, ...] = ()


class Instructions(BaseModel):
    """The rendered instructions for one turn, with what produced them."""

    model_config = ConfigDict(frozen=True)

    text: str
    activation: Activation


class SkillSyncReport(BaseModel):
    """What one start-up sync changed in procedural memory."""

    added: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    retired: list[str] = []

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.retired)
