"""Generic BasicSanity value types.

BasicSanity is deliberately narrower than business policy. It answers only
cheap, deterministic, authoritative questions that can be settled before the
agent spends an LLM/model call.
"""

from typing import Any

from pydantic import BaseModel, Field


class BasicSanityRequest(BaseModel):
    """Information available to an agent-specific sanity checker.

    `actor` is who is asking; `subject_id` is whose records the request
    concerns, as an opaque string. Deliberately not `customer_id`: the
    platform owns this contract and must not oblige every agent built on
    it to have customers, or to number them. An agent that reasons about
    patients, tenants or accounts puts its own identifier here and parses
    it back in its own checker, where that vocabulary belongs.

    Empty means no subject was established — which a checker should treat
    as a reason to stop, not as permission to proceed.
    """

    query: str
    agent_id: str = ""
    session_id: str = ""
    actor: str = ""
    subject_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BasicSanityFinding(BaseModel):
    """One deterministic sanity check and its evidence.

    ``applicable=False`` means the finding was deliberately not evaluated
    because its prerequisite was absent. This is different from a PASS: the
    checker found nothing wrong, but it did not establish the fact.
    """

    check: str
    passed: bool
    applicable: bool = True
    reason: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class BasicSanityResult(BaseModel):
    """Result of the pre-LLM sanity pass."""

    passed: bool
    reason_code: str = "BASIC_SANITY_PASSED"
    message: str = ""
    findings: list[BasicSanityFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
