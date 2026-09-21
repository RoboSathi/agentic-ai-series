"""Generic intent-classification contracts.

The platform owns classifier mechanics and neutral result types. Individual
agents own their intent taxonomy, label mapping, routing threshold, and
out-of-scope rules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class IntentDisposition(StrEnum):
    SUPPORTED = "supported"
    CLARIFY = "clarify"
    OUT_OF_SCOPE = "out_of_scope"


class IntentCandidate(BaseModel):
    """One model-native label and its score."""

    intent: str
    score: float = Field(ge=0.0, le=1.0)


class IntentAssessment(BaseModel):
    """Agent-level intent plus routing decision."""

    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    disposition: IntentDisposition
    source: str
    candidates: list[IntentCandidate] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    matched_rule: str = ""
    clarification: str = ""
    reason: str = ""
    model: str = ""
    score_type: str = "softmax_score"


@runtime_checkable
class IntentClassifierInterface(Protocol):
    """Provider-neutral contract for a single-label intent classifier."""

    @property
    def model_name(self) -> str:
        """Human-readable model identifier for logs/UI."""
        ...

    @property
    def score_type(self) -> str:
        """Describe what the returned scores mean."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the classifier weights/tokenizer are resident in memory."""
        ...

    async def ensure_loaded(self) -> None:
        """Load/download the classifier artifact once before inference."""
        ...

    def ensure_loaded_sync(self) -> None:
        """Synchronously initialize the provider before an event-loop boundary."""
        ...

    async def classify(self, text: str) -> list[IntentCandidate]:
        """Return the model's native labels sorted by descending score."""
        ...
