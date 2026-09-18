"""Sheridan automation levels (E03, LLD §11).

Sheridan & Verplank's 1978 scale, from 1 (the human does everything) to
10 (the computer acts and tells the human nothing). "Human in the loop"
names no point on it — the useful question is always *which level*, for
*which action*, at *what amount*.

Level is per action and per amount, because the same tool is a different
decision at different sizes: a $10 refund and a $10,000 refund are the
same function call and not remotely the same risk.
"""

from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.types import AutomationAssessment
from agentic_commons.logging import get_logger

# Sheridan's scale, kept for the audit record: a stored level must mean
# something to whoever reads it later.
SHERIDAN_LEVELS: dict[int, str] = {
    1: "human does everything",
    2: "computer offers alternatives",
    3: "computer narrows the alternatives",
    4: "computer suggests one alternative",
    5: "computer executes if the human approves",
    6: "computer executes unless the human vetoes in time",
    7: "computer executes then necessarily informs the human",
    8: "computer informs the human only if asked",
    9: "computer informs the human only if it decides to",
    10: "computer acts autonomously and ignores the human",
}


class AutomationLevelPolicy:
    """The automation level for an action, and whether a human is required."""

    def __init__(self) -> None:
        settings = get_settings().gatekeeper.automation
        self._enabled = settings.enabled
        self._approval_below = settings.human_approval_below
        self._levels = settings.levels
        self._escalation_amounts = settings.per_action_amount_escalation_usd
        self._logger = get_logger("gatekeeper")

    def level_for(self, action: str) -> int:
        """The configured Sheridan level for an action.

        An unconfigured action sits at level 5 — executes only with human
        approval. Defaulting upward would automate whatever nobody
        remembered to classify.
        """
        bare = action.rsplit(".", maxsplit=1)[-1]
        level = self._levels.get(bare)
        if level is None:
            self._logger.debug("No automation level configured", action=action)
            return self._approval_below
        return level

    def escalation_amount(self, action: str) -> float:
        """The amount above which this action needs a human, regardless of level."""
        bare = action.rsplit(".", maxsplit=1)[-1]
        return self._escalation_amounts.get(bare, float("inf"))

    def assess(self, action: str, amount: float = 0.0) -> AutomationAssessment:
        """Whether this action, at this amount, may proceed without a human."""
        if not self._enabled:
            return AutomationAssessment(
                action=action, level=10, requires_human=False, reason="automation policy disabled"
            )
        level = self.level_for(action)
        ceiling = self.escalation_amount(action)
        if amount > ceiling:
            return AutomationAssessment(
                action=action,
                level=level,
                requires_human=True,
                amount=amount,
                reason=f"${amount:,.2f} exceeds the ${ceiling:,.2f} auto-approval ceiling",
            )
        if level < self._approval_below:
            return AutomationAssessment(
                action=action,
                level=level,
                requires_human=True,
                amount=amount,
                reason=(
                    f"Sheridan level {level} ({SHERIDAN_LEVELS.get(level, 'unknown')}) "
                    f"is below the level-{self._approval_below} autonomy floor"
                ),
            )
        return AutomationAssessment(
            action=action,
            level=level,
            requires_human=False,
            amount=amount,
            reason=f"Sheridan level {level}: {SHERIDAN_LEVELS.get(level, 'unknown')}",
        )
