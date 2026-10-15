"""Prompt-injection value types (LLD §12.3)."""

from typing import Any

from pydantic import BaseModel, Field


class SeparatedContext(BaseModel):
    """Trusted instructions and untrusted data, kept structurally apart.

    The separation is the control. Once both are concatenated into one
    prompt, no downstream component can tell which half asked for what —
    the model certainly cannot.
    """

    instructions: str
    data: str
    delimited_data: str = ""
    suspicious_phrases: list[str] = Field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        """Whether the data half contains known injection phrasing."""
        return bool(self.suspicious_phrases)


class InjectionAssessment(BaseModel):
    """What was found in untrusted content, and what it cost the caller.

    Detection is advisory by design: it narrows what the agent may
    do next, it never decides that the content is safe.
    """

    suspicious: bool
    phrases: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    reason: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)
