"""Compensating saga (LLD §13).

A refund touches several systems that cannot share a transaction. The
saga's contract is not "all steps succeed" — it is "no step is left
half-applied": whatever committed before the failure is undone, in
reverse order, because a later step may depend on an earlier one.

Reverse order matters. Releasing a reservation before the refund that
consumed it is undone leaves the two out of step.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, logged
from agentic_commons.resilience.types import SagaResult, SagaStep


@runtime_checkable
class Saga(Protocol):
    """Runs steps, compensating what completed if a later step fails."""

    async def run(self, steps: list[SagaStep]) -> SagaResult: ...


class CompensatingSaga:
    """Forward execution with reverse compensation on failure."""

    def __init__(self) -> None:
        self._enabled = get_settings().resilience.saga.enabled
        self._logger = get_logger("resilience")

    @logged("resilience")
    async def run(self, steps: list[SagaStep]) -> SagaResult:
        """Run every step; on failure, undo what already committed.

        Disabled, the steps still run in order but nothing is unwound: a
        failure leaves whatever committed in place, which is exactly the
        half-applied state the saga exists to prevent — and the log says so.
        """
        completed: list[SagaStep] = []
        for step in steps:
            try:
                await step.action()
            except Exception as exc:
                self._logger.error("Saga step failed", step=step.name, error=str(exc))
                if not self._enabled:
                    return self._uncompensated(completed, step.name, str(exc))
                return await self._compensate(completed, step.name, str(exc))
            completed.append(step)
            self._logger.debug("Saga step committed", step=step.name)
        return SagaResult(succeeded=True, completed=[step.name for step in completed])

    def _uncompensated(self, completed: list[SagaStep], failed_step: str, error: str) -> SagaResult:
        """A failure reported without unwinding, because the saga is disabled."""
        self._logger.warning(
            "Saga disabled — committed steps left in place",
            failed_step=failed_step,
            committed=[step.name for step in completed],
        )
        return SagaResult(
            succeeded=False,
            completed=[step.name for step in completed],
            failed_step=failed_step,
            error=error,
        )

    async def _compensate(
        self, completed: list[SagaStep], failed_step: str, error: str
    ) -> SagaResult:
        """Undo committed steps in reverse, recording any that would not undo.

        A compensation that itself fails is recorded and the unwinding
        continues. Stopping there would leave *more* state half-applied
        than the original failure did — and the record is what a human
        needs to finish the job by hand.
        """
        compensated: list[str] = []
        errors: dict[str, str] = {}
        for step in reversed(completed):
            if step.compensation is None:
                continue
            try:
                await step.compensation()
                compensated.append(step.name)
            except Exception as exc:
                errors[step.name] = str(exc)
                self._logger.error(
                    "Compensation failed — manual reconciliation required",
                    step=step.name,
                    error=str(exc),
                )
        return SagaResult(
            succeeded=False,
            completed=[step.name for step in completed],
            compensated=compensated,
            failed_step=failed_step,
            error=error,
            compensation_errors=errors,
        )
