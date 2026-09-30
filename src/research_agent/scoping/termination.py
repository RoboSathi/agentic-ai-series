"""When the interview may end — enforced here, in code, never in the prompt.

Hybrid termination:
- the agent may only *propose* done, and only when no required item is
  open; a model that proposes early is overruled, not obeyed;
- the user confirms (`/done`), and may say `/done` at any time — open
  required items then become known unknowns on the Brief;
- at `max_turns` the agent stops asking, lists what is open, and offers
  `/waive <id|all>` or `/done`.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from research_agent.domain.scope import CoverageItem
from research_agent.prompts.interviewer import InterviewTurn
from research_agent.scoping.checklist import open_items

TurnOutcome = Literal["ask", "propose_done", "turn_cap"]


class TerminationDecision(BaseModel):
    outcome: TurnOutcome
    open_required: list[str] = Field(default_factory=list)
    overruled: bool = False


class TerminationPolicy:
    """Decides what an interviewer turn is allowed to become."""

    def __init__(self, max_turns: int) -> None:
        self._max_turns = max_turns

    def may_propose_done(self, items: Sequence[CoverageItem]) -> bool:
        return not open_items(items, required_only=True)

    def decide(
        self, turn: InterviewTurn, items: Sequence[CoverageItem], turn_number: int
    ) -> TerminationDecision:
        """Ask again, propose done (only if allowed), or stop at the turn cap."""
        still_open = [item.item_id for item in open_items(items, required_only=True)]
        allowed = self.may_propose_done(items)
        if turn.proposes_done and allowed:
            return TerminationDecision(outcome="propose_done")
        if turn_number >= self._max_turns:
            return TerminationDecision(outcome="turn_cap", open_required=still_open)
        return TerminationDecision(
            outcome="ask", open_required=still_open, overruled=turn.proposes_done and not allowed
        )
