"""Cost-stack aggregation (E44, LLD §6).

This is the single place cost is tracked — there is deliberately no
`economics` package. The aggregator subscribes to completed spans rather
than being called by every cost-incurring module.
"""

from typing import Protocol, runtime_checkable

from agent_platform.observability.types import CostComponent, CostStack


@runtime_checkable
class CostAggregator(Protocol):
    """Accumulates per-request cost and reports cost per successful outcome."""

    def record(self, request_id: str, component: CostComponent, cost_usd: float) -> None: ...

    def cost_per_successful_outcome(self) -> float: ...


class CostAggregatorImpl:
    """Per-request cost stacks plus the E44 cost-per-success denominator."""

    def __init__(self) -> None:
        self._stacks: dict[str, CostStack] = {}
        self._successes: set[str] = set()
        self._failures: set[str] = set()

    def record(self, request_id: str, component: CostComponent, cost_usd: float) -> None:
        """Add cost for one component of one request."""
        stack = self._stacks.setdefault(request_id, CostStack())
        setattr(stack, component, getattr(stack, component) + cost_usd)

    def record_outcome(self, request_id: str, success: bool) -> None:
        """Mark a request's eval outcome — the cost-per-success denominator."""
        (self._successes if success else self._failures).add(request_id)

    @property
    def total_cost(self) -> float:
        """Total cost across every recorded request."""
        return sum(stack.total for stack in self._stacks.values())

    def cost_per_successful_outcome(self) -> float:
        """Total spend divided by *successful* outcomes only.

        The denominator is successes, never raw call count: a cheaper but
        wrong answer is not a saving (E44).
        """
        if not self._successes:
            return 0.0
        return self.total_cost / len(self._successes)
