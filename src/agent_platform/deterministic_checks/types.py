"""Generic DeterministicCheck value types.

DeterministicCheck is deliberately narrower than business policy. It answers only
cheap, deterministic, authoritative questions that can be settled before the
agent spends an LLM/model call.
"""

from typing import Any

from pydantic import BaseModel, Field


class DeterministicCheckRequest(BaseModel):
    """Information available to an agent-specific deterministic checker.

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


class DeterministicCheckFinding(BaseModel):
    """One deterministic deterministic check and its evidence.

    ``applicable=False`` means the finding was deliberately not evaluated
    because its prerequisite was absent. This is different from a PASS: the
    checker found nothing wrong, but it did not establish the fact.
    """

    check: str
    passed: bool
    applicable: bool = True
    reason: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class DeterministicCheckResult(BaseModel):
    """Result of the pre-LLM deterministic checks."""

    passed: bool
    reason_code: str = "DETERMINISTIC_CHECKS_PASSED"
    message: str = ""
    findings: list[DeterministicCheckFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
