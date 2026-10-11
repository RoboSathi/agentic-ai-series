"""The vocabulary of the profile and skills in a prompt (ADR 0010).

A *skill* is how-to knowledge in procedural memory, held as versions. An
*authored* skill is a SKILL.md in the repository; a *learned* skill is one
a dream proposed and a curator approved. An *activation* is which skills
one turn's prompt carried, and why.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Where a skill came from: a file in the repository, or a dream that
# proposed it and a curator who approved it.
SkillSource = Literal["authored", "learned"]


class SkillDocument(BaseModel):
    """One skill's text, parsed: what it is for and what it says."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    triggers: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    # Read tools the harness runs before the first model call, and every tool the
    # skill uses (what a turn running it is offered), by model name.
    prefetch: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    # The whole skill without its frontmatter (what `/skills show` prints).
    body: str
    # What the model reads: only the configured sections (`skills.render.sections_sent`).
    prompt: str = ""
    # The skill's numbered steps, one line each (what its card lists).
    steps: tuple[str, ...] = ()
    # What is stored and digested: the file as written, references included.
    raw: str


class Skill(BaseModel):
    """A skill as procedural memory holds it: a document at a version."""

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
    """A skill chosen for one turn, and the reason it was."""

    model_config = ConfigDict(frozen=True)

    skill: Skill
    reason: str


class Activation(BaseModel):
    """Everything procedural one prompt carried."""

    model_config = ConfigDict(frozen=True)

    profile: AgentProfile
    skills: tuple[ActivatedSkill, ...] = ()

    def prefetch(self) -> list[str]:
        """Every active skill's prefetch tools, in order, each once."""
        return list(dict.fromkeys(t for item in self.skills for t in item.skill.document.prefetch))

    def tools(self) -> list[str]:
        """Every tool the active skills use, each once (empty: no narrowing)."""
        return list(dict.fromkeys(t for item in self.skills for t in item.skill.document.tools))


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
