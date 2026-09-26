"""Context value types used by agents and the common context pipeline."""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

# What a chunk is for, which determines how it may be trimmed.
ChunkKind = Literal["system", "policy", "history", "retrieved", "memory", "tool_result"]

# E17's five context failure modes.
ContextFailureMode = Literal["rot", "distraction", "confusion", "clash", "poisoning"]


class Context(BaseModel):
    """The runtime context assembled for one model request.

    The contents are intentionally untyped at this layer. Each agent builds
    whatever representation its model request needs; commons only provides
    a small container for passing that runtime context around.
    """

    items: list[Any] = Field(default_factory=list)

    def add(self, item: Any) -> None:
        """Append one item to the context."""
        self.items.append(item)

    def extend(self, items: list[Any]) -> None:
        """Append several items to the context."""
        self.items.extend(items)

    def clear(self) -> None:
        """Remove all items from the context."""
        self.items.clear()

    @property
    def text(self) -> str:
        """Render context items as a prompt fragment."""
        return "\n\n".join(str(item) for item in self.items)


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
    """The detailed output of the common context assembly pipeline."""

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


# The line `format` puts above every chunk: `[kind]` or `[kind: source]`.
# Written and recognised in one place, so a reader of assembled context
# (the failure detector) can never mistake the label for content.
_PROVENANCE_HEADER = re.compile(r"^\[[a-z_]+(?::[^\]]*)?\]$")


def provenance_header(chunk: "Chunk") -> str:
    """The label for one chunk's provenance."""
    return f"[{chunk.kind}: {chunk.source}]" if chunk.source else f"[{chunk.kind}]"


def is_provenance_header(line: str) -> bool:
    """Whether a line is a provenance label rather than content."""
    return bool(_PROVENANCE_HEADER.match(line.strip()))


class ContextFinding(BaseModel):
    """One detected failure in an assembled window, and what was done about it."""

    mode: ContextFailureMode
    source: str
    rule: str
    detail: str
    # Position of the chunk in the window, when the finding is about one.
    index: int | None = None
    # What mitigation did: "dropped", "kept (higher authority)", "flagged".
    action: str = "flagged"
