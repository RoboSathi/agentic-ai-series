"""The research plan: a DAG of sub-questions, each traced to Rubric Items."""

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.runtime.scheduling import TaskNode


class SubQuestion(BaseModel):
    """One node of the plan: a question, its queries, and what it answers for."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    question: str
    depends_on: tuple[str, ...] = ()
    # The Rubric Items this node exists to satisfy (at least one).
    rubric_items: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    # Which Brief subject (an option, claim or concept) it is about, if any.
    subject: str = ""

    def task(self) -> TaskNode:
        return TaskNode(node_id=self.node_id, depends_on=self.depends_on)


class PlanDag(BaseModel):
    """The whole plan."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[SubQuestion, ...] = ()
    rationale: str = ""

    def node(self, node_id: str) -> SubQuestion | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def tasks(self) -> list[TaskNode]:
        return [node.task() for node in self.nodes]


class PlanValidation(BaseModel):
    """The plan validation gate's findings; empty means the plan may run."""

    problems: list[str] = Field(default_factory=list)
    cycle: list[str] = Field(default_factory=list)
    # Must-pass criteria no sub-question serves. Not a reason to refuse the plan (gap analysis
    # can still reach them), but a reason to ask for one repair, and shown on the plan card.
    uncovered: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems
