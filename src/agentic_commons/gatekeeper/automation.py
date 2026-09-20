"""Sheridan automation levels (E03, LLD §11)."""

from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.types import AutomationAssessment

SHERIDAN_LEVELS = {
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
    """Operationalizes the configured Sheridan automation envelope."""

    def __init__(self) -> None:
        settings = get_settings().gatekeeper.automation
        self._enabled = settings.enabled
        self._approval_below = settings.human_approval_below
        self._levels = settings.levels
        self._escalation_amounts = settings.per_action_amount_escalation_usd

    @property
    def approval_floor(self) -> int:
        """Configured level at or below which the current implementation requires review."""
        return self._approval_below

    def level_for(self, action: str) -> int:
        """The configured Sheridan level for an action.

        An unconfigured action sits at the approval floor rather than
        defaulting upward and silently becoming autonomous.
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
        level = self.level_for(action)
        ceiling = self.escalation_amount(action)
        within_ceiling = amount <= ceiling
        level_meets_floor = level >= self._approval_below
        meaning = SHERIDAN_LEVELS.get(level, "unconfigured Sheridan level")

        if not self._enabled:
            assessment = AutomationAssessment(
                action=action,
                level=10,
                requires_human=False,
                reason="automation policy disabled",
                amount=amount,
                policy_enabled=False,
                automation_floor=self._approval_below,
                level_meets_floor=True,
                auto_approval_ceiling=ceiling,
                amount_within_ceiling=True,
                level_meaning=SHERIDAN_LEVELS[10],
            )
        elif amount > ceiling:
            assessment = AutomationAssessment(
                action=action,
                level=level,
                requires_human=True,
                amount=amount,
                reason=f"${amount:,.2f} exceeds the ${ceiling:,.2f} auto-approval ceiling",
                policy_enabled=True,
                automation_floor=self._approval_below,
                level_meets_floor=level_meets_floor,
                auto_approval_ceiling=ceiling,
                amount_within_ceiling=False,
                level_meaning=meaning,
            )
        elif level < self._approval_below:
            assessment = AutomationAssessment(
                action=action,
                level=level,
                requires_human=True,
                amount=amount,
                reason=(
                    f"Sheridan level {level} ({meaning}) is below the "
                    f"level-{self._approval_below} autonomy floor"
                ),
                policy_enabled=True,
                automation_floor=self._approval_below,
                level_meets_floor=False,
                auto_approval_ceiling=ceiling,
                amount_within_ceiling=within_ceiling,
                level_meaning=meaning,
            )
        else:
            assessment = AutomationAssessment(
                action=action,
                level=level,
                requires_human=False,
                amount=amount,
                reason=f"Sheridan level {level}: {meaning}",
                policy_enabled=True,
                automation_floor=self._approval_below,
                level_meets_floor=True,
                auto_approval_ceiling=ceiling,
                amount_within_ceiling=within_ceiling,
                level_meaning=meaning,
            )

        return assessment
