"""How one turn went (E37, LLD §6).

Eight SLOs are configured. Until this existed the agent recorded two of
them, so six error budgets could never be breached: a target nothing
observes is not a target, it is a number in a file.

This class is where a turn is scored. It is deliberately separate from
the graph: the graph decides what to *do*, and mixing "what happened"
with "how it went" into the same methods is how accounting ends up
sprinkled through eight nodes and applied in seven of them.

Tokens are counted; money is not. Pricing a turn means carrying a rate
card for every provider and a unit price for every tool, retrieval and
minute of human review — a model that is wrong the moment a price
changes, and wrong in the direction that flatters it. The token count is
the honest measure the harness actually observes.
"""

import time

from pydantic import BaseModel, Field

from agent_platform.llm.types import TokenUsage
from agent_platform.logging import get_logger, instrument
from agent_platform.observability import (
    FactRecorder,
    NullFactRecorder,
    SLORegistry,
    TurnOutcomeFact,
)

# The SLOs this agent is in a position to observe, and what each means
# here. Named as constants so a metric can never be recorded under a
# slightly different spelling than the one configured — the failure that
# makes a dashboard quietly read zero forever.
TASK_SUCCESS = "task_success_rate"
UNAUTHORIZED_ACTION = "unauthorized_action_rate"
HALLUCINATION = "hallucination_rate"
ESCALATION = "escalation_rate"
LOOP_ABORT = "loop_abort_rate"
TOOL_FAILURE = "tool_failure_rate"
RECOVERY_SUCCESS = "recovery_success_rate"
POLICY_VIOLATION = "policy_violation_rate"
# How a turn ended, as its outcome fact names it.
ANSWERED, ESCALATED, UNANSWERED = "answered", "escalated", "unanswered"


class TurnOutcome(BaseModel):
    """Everything about one turn that the SLOs and the cost stack read."""

    session_id: str
    answered: bool = False
    escalated: bool = False
    # A tool the Gatekeeper refused because the model was never offered
    # it, or because the actor's role does not carry the permission.
    unauthorized_attempts: int = 0
    # A refusal on policy grounds: a ledger threshold, a surface class, a
    # moderation flag. Distinct from unauthorized, because the actor was
    # allowed to ask and the answer was still no.
    policy_refusals: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    retrievals: int = 0
    evaluator_calls: int = 0
    llm_calls: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    hallucination_rate: float = 0.0
    # The turn spent its whole tool budget without reaching an answer.
    loop_aborted: bool = False
    # Something failed and a later attempt succeeded — the resilience
    # patterns doing their job, which is only visible if counted.
    recoveries_attempted: int = 0
    recoveries_succeeded: int = 0


@instrument("customer_support_agent")
class TurnAccountant:
    """Scores one finished turn against the SLOs and the cost model."""

    def __init__(self, slos: SLORegistry | None = None, facts: FactRecorder | None = None) -> None:
        self._slos = slos
        # The same observations, kept: the registry is this process's view,
        # the fact is what the mart computes windowed SLOs from.
        self._facts = facts or NullFactRecorder()
        self._started = time.perf_counter()
        self._logger = get_logger("customer_support_agent")

    def begin(self) -> None:
        """Start timing a turn."""
        self._started = time.perf_counter()

    def settle(self, outcome: TurnOutcome) -> None:
        """Record every SLO observation for one turn, live and as a fact."""
        self._record_slos(outcome)
        self._facts.record(self._fact(outcome))
        self._logger.info(
            "Turn settled",
            session_id=outcome.session_id,
            answered=outcome.answered,
            escalated=outcome.escalated,
            tool_calls=outcome.tool_calls,
            total_tokens=outcome.usage.total_tokens,
        )

    def _fact(self, outcome: TurnOutcome) -> TurnOutcomeFact:
        """The turn as the mart reads it; every field here is one this agent observes."""
        return TurnOutcomeFact(
            session_id=outcome.session_id,
            outcome=(
                ESCALATED if outcome.escalated else ANSWERED if outcome.answered else UNANSWERED
            ),
            latency_ms=(time.perf_counter() - self._started) * 1000,
            answered=outcome.answered,
            escalated=outcome.escalated,
            unauthorized_attempts=outcome.unauthorized_attempts,
            policy_refusals=outcome.policy_refusals,
            loop_aborted=outcome.loop_aborted,
            hallucination_rate=outcome.hallucination_rate,
            tool_calls=outcome.tool_calls,
            tool_failures=outcome.tool_failures,
            recoveries_attempted=outcome.recoveries_attempted,
            recoveries_succeeded=outcome.recoveries_succeeded,
        )

    def _record_slos(self, outcome: TurnOutcome) -> None:
        """Every SLO this agent can observe, observed once per turn.

        Rates are recorded as one boolean observation per turn rather
        than as a running ratio: the registry averages, so a turn is the
        unit, and a component that pre-averaged would make two turns
        count as one.
        """
        if self._slos is None:
            return
        self._slos.record_outcome(TASK_SUCCESS, outcome.answered)
        self._slos.record(UNAUTHORIZED_ACTION, 1.0 if outcome.unauthorized_attempts else 0.0)
        self._slos.record(POLICY_VIOLATION, 1.0 if outcome.policy_refusals else 0.0)
        self._slos.record(ESCALATION, 1.0 if outcome.escalated else 0.0)
        self._slos.record(LOOP_ABORT, 1.0 if outcome.loop_aborted else 0.0)
        if outcome.hallucination_rate:
            self._slos.record(HALLUCINATION, outcome.hallucination_rate)
        if outcome.tool_calls:
            self._slos.record(TOOL_FAILURE, outcome.tool_failures / outcome.tool_calls)
        if outcome.recoveries_attempted:
            self._slos.record(
                RECOVERY_SUCCESS,
                outcome.recoveries_succeeded / outcome.recoveries_attempted,
            )
