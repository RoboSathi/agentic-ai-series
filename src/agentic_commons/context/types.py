"""Context value types (LLD §7)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# What a chunk is for, which determines how it may be trimmed.
ChunkKind = Literal["system", "policy", "history", "retrieved", "memory", "tool_result"]

# E17's five context failure modes.
ContextFailureMode = Literal["rot", "distraction", "confusion", "clash", "poisoning"]


class Chunk(BaseModel):
    """One candidate piece of context, with what it costs to include."""

    content: str
    kind: ChunkKind = "retrieved"
    source: str = ""
    score: float = 0.0
    token_estimate: int = 0
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """What the context is being assembled for.

    Assembly needs to know the task, not just the query text: a refund
    request and a status lookup want different things in the window even
    when the words overlap.
    """

    query: str
    session_id: str = ""
    actor: str = ""
    kind: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssembledContext(BaseModel):
    """The window as assembled, plus the accounting behind it.

    `dropped` and `applied_strategies` are not diagnostics — they are the
    record of what the model was *not* shown, which is the first question
    to ask when an answer is wrong.
    """

    chunks: list[Chunk] = Field(default_factory=list)
    budget_tokens: int = 0
    used_tokens: int = 0
    dropped: list[str] = Field(default_factory=list)
    applied_strategies: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """The assembled context as a single prompt fragment."""
        return "\n\n".join(chunk.content for chunk in self.chunks)

    @property
    def within_budget(self) -> bool:
        return self.used_tokens <= self.budget_tokens


class ContextHealth(BaseModel):
    """A failure-mode assessment at one token checkpoint (E17)."""

    checkpoint_tokens: int
    observed_score: float
    baseline_score: float
    degraded: bool
    modes: list[ContextFailureMode] = Field(default_factory=list)
    detail: str = ""
