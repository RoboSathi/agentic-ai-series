"""Generic BasicSanity value types.

BasicSanity is deliberately narrower than business policy. It answers only
cheap, deterministic, authoritative questions that can be settled before the
agent spends an LLM/model call.
"""

from typing import Any

from pydantic import BaseModel, Field


class BasicSanityRequest(BaseModel):
    """Information available to an agent-specific sanity checker."""

    query: str
    agent_id: str = ""
    session_id: str = ""
    actor: str = ""
    customer_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BasicSanityFinding(BaseModel):
    """One deterministic sanity check and its evidence."""

    check: str
    passed: bool
    reason: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class BasicSanityResult(BaseModel):
    """Result of the pre-LLM sanity pass."""

    passed: bool
    reason_code: str = "BASIC_SANITY_PASSED"
    message: str = ""
    findings: list[BasicSanityFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
