"""Shared vocabulary for the pattern library (LLD §22).

Every pattern here returns a `PatternRun`. That uniformity is the point:
the patterns differ in *control flow*, not in what they produce, so
putting two of them side by side shows the only thing that actually
differs between them.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# WF: the path is fixed at design time by a person.
# AG: the path is chosen at run time by a model.
PatternFamily = Literal["WF", "AG"]

# The Three Gates (E04/E15) — what decides which family a task belongs to.
PathCardinality = Literal["single", "bounded", "open"]
BlastRadius = Literal["single_record", "account", "tenant", "global"]


class Step(BaseModel):
    """One observable step of a pattern's execution.

    Patterns record their steps rather than only their answer, because
    the trajectory *is* what distinguishes most of these patterns from
    each other — a ReAct run and a single completion can produce the
    same string by very different routes.
    """

    kind: str
    detail: str = ""
    output: str = ""
    cost_usd: float = 0.0
    tokens: int = 0


class PatternRun(BaseModel):
    """What every pattern in this library returns."""

    pattern: str
    family: PatternFamily
    output: str = ""
    steps: list[Step] = Field(default_factory=list)
    succeeded: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(step.tokens for step in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return sum(step.cost_usd for step in self.steps)

    def steps_of(self, kind: str) -> list[Step]:
        """Every step of one kind — how a test asserts a pattern's shape."""
        return [step for step in self.steps if step.kind == kind]


class PatternCard(BaseModel):
    """One row of the pattern-selection decision record (E04's exit artifact).

    The record exists so that "we chose an agent here" is a decision
    someone made against stated criteria, rather than a default.
    """

    name: str
    family: PatternFamily
    summary: str
    use_when: str
    avoid_when: str
    path_cardinality: PathCardinality
    blast_radius: BlastRadius
    certifiable: bool
    depth_episode: str
    module: str = ""

    @property
    def gates_pass(self) -> bool:
        """Whether the Three Gates permit this shape without a human in line.

        An open-ended path with a wide blast radius and no certifiable
        result is the profile that should not be automated — naming which
        gate fails turns "this feels risky" into an argument.
        """
        wide = self.blast_radius in {"tenant", "global"}
        if self.path_cardinality == "open" and wide:
            return False
        return not (not self.certifiable and (wide or self.path_cardinality == "open"))
