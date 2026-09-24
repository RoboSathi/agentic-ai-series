"""What every node group needs: state notes, halts, and the status feed.

The attribute declarations below are the interesting part. Each node
group is a mixin over the one agent, so without them there is nothing
saying which collaborators a group actually uses — and a type checker
could not tell you either. Declared here once, they are a readable
inventory of what the graph is composed of, and `mypy --strict` holds
every node group to it.
"""

from typing import Any

from agentic_commons.audit import AuditLedger
from agentic_commons.basic_sanity import BasicSanityChecker
from agentic_commons.config import FeatureSet
from agentic_commons.context import (
    ContextFailureDetector,
    FiveStagePipeline,
    StrategySelector,
)
from agentic_commons.control_loop import ToolControlLoop
from agentic_commons.evals import NollmScoringCascadeImpl
from agentic_commons.executor import (
    ToolExecutionHandler,
    ToolExecutor,
)
from agentic_commons.gatekeeper import Gatekeeper
from agentic_commons.gatekeeper.mandate import MandateIssuer
from agentic_commons.guardrails import ResponseGuardrail
from agentic_commons.hallucination_protection import DetectionLadder, GroundingReasoningDiagnoser
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import CompletionRequest, CompletionResult, Message
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability import (
    SLORegistry,
    StageReporter,
    Tracer,
)
from agentic_commons.observability.types import EventKind
from agentic_commons.resilience import (
    DeadLetterQueue,
    ResilientCaller,
)
from agentic_commons.security import SecuritySuite
from agentic_commons.security.auth.policy import AgentPolicy
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agentic_commons.verification import EffectVerifier
from customer_support_agent.control_loop import OutputQualityGate
from customer_support_agent.domain.identity import (
    Customer,
)
from customer_support_agent.graph.state import AgentState
from customer_support_agent.intent.router import CustomerSupportIntentRouter, Intent
from customer_support_agent.observability.accounting import TurnAccountant
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.presentation.orders import (
    OrderMenu,
)
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.tools.definitions import SupportTools


class NodeSupport:
    """Collaborators and small helpers shared by every node group."""

    # Injected by `CustomerSupportAgent.__init__`; declared here so each
    # mixin type-checks against the agent it is mixed into.
    _llm: LLMClient
    _gatekeeper: Gatekeeper | None
    _audit: AuditLedger
    _registry: NamespacedToolRegistry
    _tools: SupportTools
    _executor: ToolExecutor
    _handler: ToolExecutionHandler | None
    _security: SecuritySuite
    _memory: MemoryFacade | None
    _context: FiveStagePipeline | None
    _selector: ToolRetrievalSelector | None
    _cascade: NollmScoringCascadeImpl | None
    _reporter: StageReporter | None
    _resilient: ResilientCaller
    _dead_letter_queue: DeadLetterQueue | None
    _tracer: Tracer | None
    _slos: SLORegistry | None
    _grounding: DetectionLadder | None
    _verifier: EffectVerifier | None
    _mandate_issuer: MandateIssuer | None
    _agent_policy: AgentPolicy
    _features: FeatureSet
    _intents: CustomerSupportIntentRouter
    _authenticator: CustomerAuthenticator | None
    _orders: OrdersRepository | None
    _basic_sanity: BasicSanityChecker | None
    _deliberation_markers: list[str]
    _max_rounds: int
    _actor: str
    _strategies: StrategySelector
    _context_health: ContextFailureDetector
    _diagnoser: GroundingReasoningDiagnoser
    _response_guardrail: ResponseGuardrail
    _sessions: dict[str, Customer]
    _menus: dict[str, OrderMenu]
    _history_turns: int
    _conversations: dict[str, list[Message]]
    _authoritative_history: dict[str, list[str]]
    _last_supported_intent: dict[str, Intent]
    _control_loop: ToolControlLoop
    _deliberation: OutputQualityGate
    _accountant: TurnAccountant
    _logger: Any

    # The nodes a graph is wired from. Declared here rather than left
    # implicit because `edges.py` registers them by name and would
    # otherwise be the only place that knows the set exists — and a node
    # renamed in its own module would fail at graph-compile time rather
    # than at type-check time. Each is implemented by exactly one group.

    async def authenticate(self, state: AgentState) -> dict[str, Any]:
        """Establish who is asking, deterministically."""
        raise NotImplementedError

    async def choose_order(self, state: AgentState) -> dict[str, Any]:
        """Resolve which order the customer means."""
        raise NotImplementedError

    async def basic_sanity(self, state: AgentState) -> dict[str, Any]:
        """Run this agent's own pre-flight checks."""
        raise NotImplementedError

    async def prompt_injection(self, state: AgentState) -> dict[str, Any]:
        """Screen the request for an injection attempt."""
        raise NotImplementedError

    async def content_moderation(self, state: AgentState) -> dict[str, Any]:
        """Screen the request against the content policy."""
        raise NotImplementedError

    async def classify_intent(self, state: AgentState) -> dict[str, Any]:
        """Decide what is being asked, and what that permits."""
        raise NotImplementedError

    async def read_intent(self, state: AgentState) -> dict[str, Any]:
        """Answer a read-shaped intent without calling a model."""
        raise NotImplementedError

    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window the model is handed."""
        raise NotImplementedError

    async def reason(self, state: AgentState) -> dict[str, Any]:
        """Ask the model what to do next."""
        raise NotImplementedError

    async def act(self, state: AgentState) -> dict[str, Any]:
        """Execute what the model proposed, under governance."""
        raise NotImplementedError

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """Turn what happened into a reply a customer can read."""
        raise NotImplementedError

    # Helpers one group needs from another. Declaring them keeps the
    # coupling between groups visible and checked: a group that grows a
    # dependency on a sibling has to say so here.

    def _system_prompt(self) -> str:
        """The system prompt this turn is run under."""
        raise NotImplementedError

    def _prompt_messages(self, state: AgentState) -> list[Message]:
        """The transcript handed to the model this turn."""
        raise NotImplementedError

    async def _call_model(
        self, request: CompletionRequest, state: AgentState, key: str
    ) -> CompletionResult:
        """One model call, under timeout, retry and circuit breaker."""
        raise NotImplementedError

    async def _invoke(
        self, name: str, arguments: str, state: AgentState, offered: set[str]
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Run one proposed tool call through governance and execution."""
        raise NotImplementedError

    def _offer_to_memory(self, state: AgentState) -> list[str]:
        """Write what this turn established into memory."""
        raise NotImplementedError

    def _resolve_read_order_id(self, state: AgentState) -> int | None:
        """Which order a read-shaped request is about, if any."""
        raise NotImplementedError

    def history_for(self, session_id: str) -> list[Message]:
        """The transcript this session is carrying."""
        raise NotImplementedError

    def _note(self, state: AgentState, key: str, value: str, category: str = "") -> None:
        """Hold a fact this turn established in the session's window.

        Working memory is what stops the next turn re-deriving what this
        one already knows — which order the customer selected, what a
        tool returned, who they are. It is read back in
        `assemble_context`, so anything noted here is in front of the
        model on the following turn without another database read.

        Silent when no memory is wired: memory is optional in this
        graph, and an agent without it must still answer.
        """
        if self._memory is None or not state.session_id:
            return
        result = self._memory.note(
            key=key,
            value=value,
            session_id=state.session_id,
            source=self._actor,
            category=category,
        )
        self._done(
            "working_memory_write",
            event_kind="memory",
            summary="Working memory updated",
            fields={
                "operation": "write",
                "key": key,
                "category": category or "unspecified",
                "written": result.written,
                "route": result.route,
            },
        )

    def _halt(self, state: AgentState, response: str, decision: str) -> dict[str, Any]:
        """Answer now, without reaching the model."""
        return {
            "response": response,
            "decisions": [*state.decisions, decision],
            "metadata": {**state.metadata, "halted": "answered_from_db"},
        }

    def _record_slo(self, metric: str, value: float) -> None:
        """Feed one observation to the SLO registry, when one is wired."""
        if self._slos is not None:
            self._slos.record(metric, value)

    def _report(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        """Emit a stage-started event for the TUI, when one is listening."""
        if self._reporter is not None:
            self._reporter.started(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _done(
        self,
        stage: str,
        tokens: int | None = None,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        """Emit a stage-completed event with safe semantic fields."""
        if self._reporter is not None:
            self._reporter.completed(
                stage,
                tokens_delta=tokens,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _done_failed(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        """Emit a stage-failed event so the feed shows what broke."""
        if self._reporter is not None:
            self._reporter.failed(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _reporter_blocked(
        self,
        stage: str,
        *,
        detail: str | None = None,
        summary: str | None = None,
        event_kind: EventKind = "decision",
        fields: dict[str, Any] | None = None,
    ) -> None:
        if self._reporter is not None:
            self._reporter.blocked(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _reporter_warning(
        self,
        stage: str,
        *,
        detail: str | None = None,
        summary: str | None = None,
        event_kind: EventKind = "system",
        fields: dict[str, Any] | None = None,
    ) -> None:
        if self._reporter is not None:
            self._reporter.warning(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _reporter_waiting(
        self,
        stage: str,
        *,
        detail: str | None = None,
        summary: str | None = None,
        event_kind: EventKind = "stage",
        fields: dict[str, Any] | None = None,
    ) -> None:
        if self._reporter is not None:
            self._reporter.waiting(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )
