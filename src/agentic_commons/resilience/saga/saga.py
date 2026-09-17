"""Saga pattern stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.resilience.saga.types import SagaResult, SagaStep


class Saga:
    """Saga pattern stub."""

    @log_call("resilience")
    def execute(self, steps: list[SagaStep]) -> SagaResult:
        """Phase 0 stub — no real saga execution yet."""
        return SagaResult(completed=False, steps_executed=0)
