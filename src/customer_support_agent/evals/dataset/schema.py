"""The Customer Support golden case — this agent's reading of the opaque payload.

The platform's `GoldenCase` carries `payload` as a plain dict. This module
is the agent's own schema for it: who the customer is, what outcome the
written policy requires (the money, the order's state, the standing of the
action), which tools may not act, and — in prose, for the DeepEval judge —
the scenario and the outcome a good reply delivers.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from agent_platform.evals.core.contracts import GoldenCase, LabelBasis

Outcome = Literal[
    "refunded", "queued_for_review", "refused", "answered", "clarification", "out_of_scope"
]
SideEffectClass = Literal["refund", "review", "none"]

# The business result each outcome implies. Only two outcomes move money
# (or ask a person to); every other outcome leaves the books untouched.
OUTCOME_EFFECT: dict[str, SideEffectClass] = {
    "refunded": "refund",
    "queued_for_review": "review",
    "refused": "none",
    "answered": "none",
    "clarification": "none",
    "out_of_scope": "none",
}


class ExpectedRefund(BaseModel):
    """The refund (or review request) a case must produce, exactly."""

    order_id: int
    amount_cents: int


class CsaExpectation(BaseModel):
    """What the written policy says the agent must do for one case.

    `refund` names the order and amount of the refund (or the review) the case
    must produce; `order_status` is that order's state afterwards when it must
    change (a cancellation); `standing` is how the money-moving call must stand
    (VOIDABLE for a direct refund, VALID for a cancellation, RATIFIABLE when a
    person must decide).
    """

    outcome: Outcome
    refund: ExpectedRefund | None = None
    order_status: str | None = None
    standing: str | None = None
    reason_code: str | None = None
    tools_forbidden: list[str] = Field(default_factory=list)
    reply_must_mention: list[str] = Field(default_factory=list)
    # What the reply must never say (the refund limit, which is never disclosed).
    reply_must_not_mention: list[str] = Field(default_factory=list)

    @property
    def effect(self) -> SideEffectClass:
        return OUTCOME_EFFECT[self.outcome]


class CsaCasePayload(BaseModel):
    """The agent-owned half of a golden case."""

    customer: str
    expected: CsaExpectation
    # For the DeepEval judge: the situation, and what a good outcome looks like (golden
    # cases always say; a red-team attack, graded only by assertions, need not).
    scenario: str = ""
    expected_outcome: str = ""
    label_basis: LabelBasis = "written_policy"

    @classmethod
    def of(cls, case: GoldenCase) -> "CsaCasePayload":
        return cls.model_validate(case.payload)


class _CaseEntry(BaseModel):
    id: str
    category: str
    tags: list[str] = Field(default_factory=list)
    customer: str
    turns: list[str]
    scenario: str
    expected_outcome: str
    expected: CsaExpectation
    label_basis: LabelBasis = "written_policy"


class _DatasetFile(BaseModel):
    suite: str
    cases: list[_CaseEntry]


def load_golden_cases(path: Path) -> list[GoldenCase]:
    """Parse and validate the dataset file into platform golden cases."""
    dataset = _DatasetFile.model_validate(yaml.safe_load(path.read_text()))
    return [
        GoldenCase(
            suite=dataset.suite,
            case_id=entry.id,
            category=entry.category,
            tags=entry.tags,
            turns=entry.turns,
            payload=CsaCasePayload(
                customer=entry.customer,
                expected=entry.expected,
                scenario=entry.scenario,
                expected_outcome=entry.expected_outcome,
                label_basis=entry.label_basis,
            ).model_dump(mode="json"),
        )
        for entry in dataset.cases
    ]
