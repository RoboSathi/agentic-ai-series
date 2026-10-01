"""The Rubric and the Research Contract.

A Rubric Item is an acceptance criterion with a named check behind it.
The Research Contract (Brief + Rubric + Budget) is the only thing the
autonomous phase receives, and its canonical hash is recorded when the
user approves it — so "the run changed what it was asked to do" is a hash
comparison, not a judgement call.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_commons.hashing import canonical_hash
from agentic_commons.observability.types import utc_now
from agentic_commons.verification.acceptance import CheckKind, CheckSpec
from research_agent.domain.budget import Budget
from research_agent.domain.scope import ScopeBrief

ItemOrigin = Literal["template", "criterion", "known_unknown", "source_policy", "user"]


class RubricCheck(BaseModel):
    """Which named check decides a Rubric Item, and with what parameters."""

    model_config = ConfigDict(frozen=True)

    kind: CheckKind
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class RubricItem(BaseModel):
    """One acceptance criterion."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    text: str
    check: RubricCheck
    threshold: float
    must_pass: bool
    # Its share of the audit's weighted score (a must-pass item also vetoes).
    weight: float = Field(default=1.0, gt=0)
    origin: ItemOrigin
    # What in the Brief this item answers to: a criterion, subject, unknown…
    maps_to: tuple[str, ...] = ()

    def spec(self) -> CheckSpec:
        return CheckSpec(
            item_id=self.item_id,
            kind=self.check.kind,
            name=self.check.name,
            # The item's wording travels with the spec: a judge scores the criterion as written.
            params={**self.check.params, "text": self.text},
            threshold=self.threshold,
            must_pass=self.must_pass,
        )


class Rubric(BaseModel):
    """The acceptance criteria for one research question."""

    model_config = ConfigDict(frozen=True)

    items: tuple[RubricItem, ...] = ()

    def item(self, item_id: str) -> RubricItem | None:
        return next((item for item in self.items if item.item_id == item_id), None)

    @property
    def ids(self) -> list[str]:
        return [item.item_id for item in self.items]

    def specs(self) -> list[CheckSpec]:
        return [item.spec() for item in self.items]


class Contract(BaseModel):
    """Brief + Rubric + Budget, frozen at approval and identified by its hash."""

    model_config = ConfigDict(frozen=True)

    contract_id: str
    brief: ScopeBrief
    rubric: Rubric
    budget: Budget
    contract_hash: str
    frozen_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def freeze(
        cls, contract_id: str, brief: ScopeBrief, rubric: Rubric, budget: Budget
    ) -> "Contract":
        return cls(
            contract_id=contract_id,
            brief=brief,
            rubric=rubric,
            budget=budget,
            contract_hash=contract_hash(brief, rubric, budget),
        )

    def verify(self) -> bool:
        """Whether the contract still hashes to what was approved."""
        return contract_hash(self.brief, self.rubric, self.budget) == self.contract_hash


def contract_hash(brief: ScopeBrief, rubric: Rubric, budget: Budget) -> str:
    return canonical_hash(
        {
            "brief": brief.model_dump(mode="json"),
            "rubric": rubric.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
        }
    )
