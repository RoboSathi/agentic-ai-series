"""What a side effect was meant to do, and what it actually did."""

from typing import Any

from pydantic import BaseModel, Field


class IntendedEffect(BaseModel):
    """The change a tool call was supposed to make.

    `target` names the thing that should have changed — an order id, a
    document key, a row — and `expected` is what it should now look like.
    Deliberately a plain mapping: the platform compares values and has no
    opinion about what an order or a refund is.
    """

    action: str
    target: str
    expected: dict[str, Any] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    """Whether the world matches what the action claimed to have done."""

    verified: bool
    effect: IntendedEffect
    observed: dict[str, Any] = Field(default_factory=dict)
    mismatches: list[str] = Field(default_factory=list)
    reason_code: str = "EFFECT_VERIFIED"
    reason: str = ""

    @property
    def summary(self) -> str:
        """One line for a log, an audit entry or the status feed."""
        if self.verified:
            return f"{self.effect.action} on {self.effect.target} verified"
        return f"{self.effect.action} on {self.effect.target}: {'; '.join(self.mismatches)}"


__all__ = ["IntendedEffect", "VerificationReport"]
