"""Provider-neutral guard-model value types."""

from typing import Any, Literal

from pydantic import BaseModel, Field

GuardDecision = Literal["allow", "flagged", "unavailable"]


class GuardVerdict(BaseModel):
    """One safety-model classification, independent of provider or model."""

    decision: GuardDecision
    flagged: bool = False
    score: float | None = None
    categories: list[str] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    reason: str = ""
    error_code: str = ""
    http_status: int | None = None
    endpoint: str = ""
    api_called: bool = False
    latency_ms: float | None = None
    # The classifier's reply, verbatim: what the card, the log and the quarantine entry show,
    # so a flag can always be traced to what the model actually said.
    output: str = ""
    raw: Any = None

    @property
    def available(self) -> bool:
        """Whether the remote guard returned a usable classification."""
        return self.decision != "unavailable"
