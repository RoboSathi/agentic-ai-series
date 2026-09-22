"""Content-moderation value types (LLD §12.4)."""

from typing import Any

from pydantic import BaseModel, Field


class ModerationVerdict(BaseModel):
    """An advisory judgement on content.

    `flagged` is deliberately not called `blocked`. This is a Tier-2
    signal: it annotates, it escalates, it never decides. The type name
    is the enforcement — a caller cannot mistake this for a verdict with
    veto power.
    """

    flagged: bool = False
    categories: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    reason: str = ""
    advisory_only: bool = True
    checks: list[dict[str, Any]] = Field(default_factory=list)
