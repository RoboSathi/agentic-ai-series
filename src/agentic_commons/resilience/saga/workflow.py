"""Saga workflow — the reusable half of a business saga (Template Method).

Every money-moving or state-moving saga an agent writes has the same
skeleton, and only one part of it is actually the agent's:

    precondition  →  idempotency replay  →  run steps  →  remember outcome
    (agent's)        (platform)             (platform,     (platform)
                                            agent's steps)

Written out by hand in each agent, the skeleton drifts: one saga forgets
to re-check its preconditions, another remembers failed outcomes and so
replays a failure forever. Here the skeleton is fixed once and an agent
supplies only what is genuinely its own — the steps, the key that makes
a retry recognisable, and the rule that refuses to start.

The collaborators are the `Saga` and `IdempotencyGuard` Protocols, not
their concrete classes, so a workflow can run on any engine or store.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agentic_commons.logging import get_logger
from agentic_commons.resilience.idempotency.guard import IdempotencyGuard
from agentic_commons.resilience.saga.saga import Saga
from agentic_commons.resilience.types import SagaResult, SagaStep

RequestT = TypeVar("RequestT")


class SagaWorkflow(ABC, Generic[RequestT]):
    """A saga whose steps an agent defines and whose skeleton the platform owns."""

    # The `failed_step` a refusal is reported under. Nothing committed, so
    # nothing is compensated — which is what distinguishes refusing from
    # rolling back.
    PRECONDITION_STEP = "precondition"

    def __init__(self, saga: Saga, idempotency: IdempotencyGuard | None = None) -> None:
        self._saga = saga
        self._idempotency = idempotency
        self._workflow_logger = get_logger("resilience")

    async def execute(self, request: RequestT) -> SagaResult:
        """Refuse, replay, or run — in that order, for every workflow."""
        reason = self.precondition(request)
        if reason is not None:
            return self._refused(reason)
        if self._idempotency is None:
            return await self._run(request)
        key = self.idempotency_key(request)
        recorded = self._idempotency.recorded_result(key)
        if recorded is not None:
            self._workflow_logger.info("Replaying prior saga outcome", key=key)
            return SagaResult.model_validate(recorded)
        result = await self._run(request)
        if result.succeeded:
            # Only successes are remembered: a replayed failure would make
            # a transient fault permanent for that key.
            self._idempotency.remember(key, result.model_dump(mode="json"))
        return result

    @abstractmethod
    def steps(self, request: RequestT) -> list[SagaStep]:
        """The ordered steps, each paired with the action that undoes it."""

    @abstractmethod
    def idempotency_key(self, request: RequestT) -> str:
        """The key under which a retried request replays its first outcome."""

    def precondition(self, request: RequestT) -> str | None:
        """Why this request must not start, or None to proceed.

        The default admits everything. Override it wherever the step that
        moves state is the wrong place to assume a caller checked first.
        """
        return None

    async def _run(self, request: RequestT) -> SagaResult:
        return await self._saga.run(self.steps(request))

    def _refused(self, reason: str) -> SagaResult:
        self._workflow_logger.warning("Saga refused before any step ran", reason=reason)
        return SagaResult(succeeded=False, failed_step=self.PRECONDITION_STEP, error=reason)
