"""The archetype stop rule (Strategy): has this run done what its archetype asks?

Each archetype ends on a different condition — every option scored on
every criterion, no new entities in the last batch, every claim decided,
every concept covered. The rule is one small class per archetype, chosen
by the Brief's archetype; budgets and cancellation are checked separately
by the reflector, so a rule only ever answers "is the work complete?".
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_platform.runtime.verification.acceptance import CheckVerdict


class StopInput(BaseModel):
    """What a stop rule may look at after a batch."""

    verdicts: list[CheckVerdict] = Field(default_factory=list)
    entities_before: int = 0
    entities_after: int = 0


class StopDecision(BaseModel):
    stop: bool
    reason: str


@runtime_checkable
class StopRule(Protocol):
    name: str

    def decide(self, observed: StopInput) -> StopDecision: ...


def all_passed(verdicts: list[CheckVerdict], check_name: str) -> tuple[bool, int]:
    """Whether every verdict of one check passed, and how many there were."""
    relevant = [verdict for verdict in verdicts if verdict.name == check_name]
    return all(verdict.passed for verdict in relevant), len(relevant)
