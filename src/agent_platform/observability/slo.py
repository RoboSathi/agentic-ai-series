"""Agent SLOs and error budgets (E37, LLD §6).

The E37 metric list is configuration, not code: adding an SLO is a config
change, and each metric's target is what the error budget is measured
against.
"""

from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.observability.types import SLOMeasurement, SLOStatus

# Metrics where a *lower* observed value is better (rates of bad outcomes).
# Public: the analytics mart computes error budgets with the same rule.
LOWER_IS_BETTER = frozenset(
    {
        "unauthorized_action_rate",
        "hallucination_rate",
        "escalation_rate",
        "loop_abort_rate",
        "tool_failure_rate",
        "policy_violation_rate",
    }
)


@runtime_checkable
class SLOTracker(Protocol):
    """Records SLO observations and reports remaining error budget."""

    def record(self, metric: str, value: float) -> None: ...

    def error_budget_remaining(self, slo_name: str) -> float: ...


class SLORegistry:
    """One query surface over every configured SLO (Facade)."""

    def __init__(self) -> None:
        self._targets = get_settings().observability.slo_targets
        self._measurements: dict[str, list[SLOMeasurement]] = {}
        self._logger = get_logger("observability")

    def record(self, metric: str, value: float) -> None:
        """Record one observation, and say so when the budget is spent."""
        if metric not in self._targets:
            self._logger.warning("Recording an SLO with no configured target", metric=metric)
        self._measurements.setdefault(metric, []).append(SLOMeasurement(metric=metric, value=value))
        self._logger.debug("SLO observation", metric=metric, value=value)
        if metric in self._targets and not self.status(metric).within_target:
            # At WARNING because an exhausted error budget is an
            # operational event, not a statistic: somebody decides
            # whether to keep shipping.
            self._logger.warning(
                "SLO outside its target",
                metric=metric,
                observed=round(self.observed(metric), 4),
                target=self._targets[metric],
                budget_remaining=round(self.error_budget_remaining(metric), 3),
            )

    def record_outcome(self, metric: str, success: bool) -> None:
        """Record a boolean outcome as a 1.0/0.0 observation."""
        self.record(metric, 1.0 if success else 0.0)

    def observed(self, metric: str) -> float:
        """Mean observed value for a metric (0.0 when never recorded)."""
        values = [m.value for m in self._measurements.get(metric, [])]
        return sum(values) / len(values) if values else 0.0

    def error_budget_remaining(self, slo_name: str) -> float:
        """Fraction of the error budget still available, in [0, 1].

        For a "lower is better" rate the budget is the target itself; for a
        success rate the budget is the allowed shortfall (1 - target).
        """
        target = self._targets.get(slo_name)
        if target is None:
            return 1.0
        observed = self.observed(slo_name)
        if slo_name in LOWER_IS_BETTER:
            if target == 0:
                return 1.0 if observed == 0 else 0.0
            return max(0.0, min(1.0, (target - observed) / target))
        budget = 1.0 - target
        if budget <= 0:
            return 1.0 if observed >= target else 0.0
        return max(0.0, min(1.0, 1.0 - (target - observed) / budget))

    def status(self, metric: str) -> SLOStatus:
        """Current standing of one SLO."""
        target = self._targets.get(metric, 0.0)
        observed = self.observed(metric)
        within = observed <= target if metric in LOWER_IS_BETTER else observed >= target
        return SLOStatus(
            metric=metric,
            target=target,
            observed=observed,
            sample_count=len(self._measurements.get(metric, [])),
            error_budget_remaining=self.error_budget_remaining(metric),
            within_target=within,
        )

    def all_statuses(self) -> list[SLOStatus]:
        """Standing of every configured SLO."""
        return [self.status(metric) for metric in self._targets]

    def breaching(self) -> list[SLOStatus]:
        """Only the SLOs currently outside their target."""
        return [status for status in self.all_statuses() if not status.within_target]
