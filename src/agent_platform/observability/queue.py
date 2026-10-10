"""Human-review queue capacity via Little's Law (E38, LLD §6).

L = λW. Escalating more work to humans is a staffing decision with a
number attached, not a hand-wave — this is how that number is computed.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_platform.config import get_settings


class QueueCapacity(BaseModel):
    """Reviewer capacity required to keep a queue stable."""

    arrival_rate_per_hr: float
    service_time_min: float
    reviewer_hours_per_hour: float
    reviewers_required: int


@runtime_checkable
class ReviewQueueModel(Protocol):
    """Computes the review capacity an escalation rate implies."""

    def required_capacity(self, arrival_rate_per_hr: float, service_time_min: float) -> float: ...


class LittlesLawQueueModel:
    """L = λW, expressed as reviewer-hours needed per wall-clock hour."""

    def __init__(self) -> None:
        self._minutes_per_hour = get_settings().observability.review_queue.minutes_per_hour

    def required_capacity(self, arrival_rate_per_hr: float, service_time_min: float) -> float:
        """Reviewer-hours of work arriving per hour.

        E38's worked example: 100 cases/hr × 2 min ⇒ 3.33 reviewer-hours per
        hour, i.e. more than three full-time reviewers just to break even.
        """
        return arrival_rate_per_hr * (service_time_min / self._minutes_per_hour)

    def capacity_for(self, arrival_rate_per_hr: float, service_time_min: float) -> QueueCapacity:
        """Full capacity picture, including whole reviewers needed."""
        hours = self.required_capacity(arrival_rate_per_hr, service_time_min)
        return QueueCapacity(
            arrival_rate_per_hr=arrival_rate_per_hr,
            service_time_min=service_time_min,
            reviewer_hours_per_hour=hours,
            reviewers_required=int(hours) + (1 if hours % 1 else 0),
        )
