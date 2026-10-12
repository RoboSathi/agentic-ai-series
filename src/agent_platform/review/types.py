"""The vocabulary of human review: one queue, every kind of decision a person makes.

A **review item** is anything the harness stopped and handed to a person:
a refund above the agent's authority, a reply the hallucination check
cut, a message security quarantined, a fact or a skill memory would not
take on its own word. Each has a **kind**, which names what approving or
denying it does; a **subject** (whose it is — empty for the agent's own);
and, once decided, who decided, when, and the **comment** they had to give.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_platform.core.primitives.clock import utc_now

ReviewStatus = Literal["pending", "approved", "denied", "failed"]
ReviewDecision = Literal["approve", "deny"]


class ReviewItem(BaseModel):
    """One thing waiting on (or decided by) a person."""

    id: int | None = None
    agent_id: str = ""
    kind: str
    status: ReviewStatus = "pending"
    session_id: str = ""
    # Whose it is (a customer's id, say); empty for the agent's own (a skill, a quarantine).
    subject_id: str = ""
    # One line a list shows.
    summary: str = ""
    # The kind's own fields, as its handler reads them.
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    decided_by: str | None = None
    decided_at: datetime | None = None
    comment: str | None = None


class ReviewOutcome(BaseModel):
    """What a decision did: its new status, and what the reviewer is told."""

    item_id: int
    kind: str
    status: ReviewStatus
    lines: list[str] = Field(default_factory=list)
    # Anything the agent should act on afterwards (a follow-up to a customer, say).
    effects: dict[str, Any] = Field(default_factory=dict)
