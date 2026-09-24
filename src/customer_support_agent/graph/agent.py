"""The refund agent: a graph, and the subsystems its nodes call (LLD §21).

Nodes, in order:

    authenticate → choose_order → screening → classify_intent →
    assemble_context → reason ⇄ act → finalize

The `reason ⇄ act` cycle is the **Tool-Use Loop**, bounded by
`max_tool_iterations`. It is not ReAct: the model proposes a typed tool
call rather than a natural-language thought, and the harness — not the
model — decides whether that call happens.

Every node calls into `agentic_commons` and owns no platform logic of its
own. That is the invariant this package exists to hold: the running
product is a *composition*, and a second LLM client or a second auth check
appearing here would be the duplication regression found and fixed once
already.

This module holds only what belongs to the agent as a whole — its
dependencies, one turn's lifecycle, and the conversation it is carrying.
Each node group lives in `nodes/`, and the wiring in `edges.py`, because
one class with sixty methods is not a design, it is a file. The groups are
mixins over this one object rather than separate collaborators: they share
a single injected set of dependencies, and `nodes/_support.py` declares
that set once so `mypy --strict` holds every group to it.

The dependencies arrive through `build_agent(...)`. Nothing is constructed
inline, so a test can substitute a scripted LLM and an in-memory database
without patching a single import.
"""



from agentic_commons.audit import AuditLedger, NullAuditLedger
from agentic_commons.basic_sanity import BasicSanityChecker
from agentic_commons.config import FeatureSet, get_settings
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
from agentic_commons.llm.types import Message, TokenUsage
from agentic_commons.logging import get_logger
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability import (
    SLORegistry,
    StageReporter,
    Tracer,
)
from agentic_commons.resilience import (
    DeadLetterQueue,
    ResilientCaller,
)
from agentic_commons.security import SecuritySuite
from agentic_commons.security.auth.policy import AgentPolicy
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agentic_commons.verification import EffectVerifier
from customer_support_agent.config import get_agent_settings
from customer_support_agent.control_loop import IndependentScorer, OutputQualityGate
from customer_support_agent.domain.identity import (
    Customer,
    current_customer,
    set_current_customer,
)
from customer_support_agent.graph.edges import GraphWiring
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.nodes.act import ToolUseNodes
from customer_support_agent.graph.nodes.authenticate import AuthenticationNodes
from customer_support_agent.graph.nodes.context import ContextNodes
from customer_support_agent.graph.nodes.finalize import FinalizeNodes
from customer_support_agent.graph.nodes.intent import IntentNodes
from customer_support_agent.graph.nodes.reason import ReasoningNodes
from customer_support_agent.graph.nodes.screening import ScreeningNodes
from customer_support_agent.graph.state import AgentState
from customer_support_agent.intent.router import CustomerSupportIntentRouter, Intent
from customer_support_agent.observability.accounting import TurnAccountant, TurnOutcome
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.presentation.orders import (
    OrderMenu,
)
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.tools.definitions import SupportTools


class CustomerSupportAgent(
    AuthenticationNodes,
    ScreeningNodes,
    IntentNodes,
    ContextNodes,
    ReasoningNodes,
    ToolUseNodes,
    FinalizeNodes,
    GraphWiring,
    NodeSupport,
):
    """The refund agent: a graph plus the subsystems its nodes call."""

    def __init__(
        self,
        llm: LLMClient,
        gatekeeper: Gatekeeper | None,
        registry: NamespacedToolRegistry,
        tools: SupportTools,
        executor: ToolExecutor,
        security: SecuritySuite,
        memory: MemoryFacade | None = None,
        context: FiveStagePipeline | None = None,
        selector: ToolRetrievalSelector | None = None,
        cascade: NollmScoringCascadeImpl | None = None,
        reporter: StageReporter | None = None,
        resilient: ResilientCaller | None = None,
        dead_letter_queue: DeadLetterQueue | None = None,
        tracer: Tracer | None = None,
        slos: SLORegistry | None = None,
        deliberation: OutputQualityGate | None = None,
        grounding: DetectionLadder | None = None,
        authenticator: CustomerAuthenticator | None = None,
        orders: OrdersRepository | None = None,
        basic_sanity: BasicSanityChecker | None = None,
        agent_policy: AgentPolicy | None = None,
        intent_router: CustomerSupportIntentRouter | None = None,
        features: FeatureSet | None = None,
        handler: ToolExecutionHandler | None = None,
        audit: AuditLedger | None = None,
        verifier: EffectVerifier | None = None,
        mandate_issuer: MandateIssuer | None = None,
    ) -> None:
        settings = get_agent_settings()
        # Which stages this agent has. Read once, here, because the graph
        # is compiled from it — a feature flipped afterwards reaches a
        # running agent only by rebuilding one.
        self._features = features if features is not None else get_settings().features
        self._llm = llm
        self._gatekeeper = gatekeeper
        # Written by the Gatekeeper when there is one, and by the agent
        # itself when there is not — an ungoverned execution is exactly
        # the one the trail must not be missing.
        self._audit = audit or NullAuditLedger()
        self._registry = registry
        self._tools = tools
        self._executor = executor
        # The raw domain adapter, used only where the governed execution
        # boundary has been withheld. See `_execute_ungoverned`.
        self._handler = handler
        self._security = security
        self._memory = memory
        self._context = context
        self._selector = selector
        self._cascade = cascade
        self._reporter = reporter
        # The model is a remote dependency like any other: bounded by a
        # timeout, retried under an idempotency key, and behind a breaker
        # so a provider outage fails fast instead of queueing.
        self._resilient = resilient or ResilientCaller("llm")
        self._dead_letter_queue = dead_letter_queue
        self._tracer = tracer
        self._slos = slos
        self._grounding = grounding
        # Reads an effect back after execution. A result it cannot
        # verify never becomes authoritative, so no customer-facing
        # claim can rest on an action that did not actually land.
        self._verifier = verifier
        # Says what this turn's request authorizes. Without one the
        # Gatekeeper reports that the question was never asked,
        # rather than quietly treating silence as consent.
        self._mandate_issuer = mandate_issuer
        self._agent_policy = agent_policy
        if self._agent_policy is None:
            raise ValueError("CustomerSupportAgent requires an AgentPolicy loaded from persistence")
        self._deliberation_markers = [m.casefold() for m in settings.deliberation_markers]
        self._max_rounds = settings.max_tool_iterations
        self._actor = self._agent_policy.agent_id
        self._strategies = StrategySelector()
        # The window the model is handed is measured, not assumed: five
        # failure modes, checked on what was actually assembled.
        self._context_health = ContextFailureDetector()
        self._diagnoser = GroundingReasoningDiagnoser()
        self._response_guardrail = ResponseGuardrail(self._grounding)
        if intent_router is None:
            raise ValueError("CustomerSupportAgent requires an intent_router from the composition root")
        self._intents = intent_router
        self._authenticator = authenticator
        self._orders = orders
        self._basic_sanity = basic_sanity
        # Who each session has proved itself to be. Authentication is per
        # conversation: a customer verifies once and stays verified for
        # that session, rather than retyping a code every turn.
        self._sessions: dict[str, Customer] = {}
        # The order list each session was last shown, so a bare "2"
        # resolves back to an order.
        self._menus: dict[str, OrderMenu] = {}
        self._history_turns = settings.history_turns
        self._conversations: dict[str, list[Message]] = {}
        # Tool observations are not customer dialogue. Keep a bounded,
        # separate history so follow-up turns can resolve references from
        # authoritative facts without treating an old action as a new one.
        self._authoritative_history: dict[str, list[str]] = {}
        self._last_supported_intent: dict[str, Intent] = {}
        self._control_loop = ToolControlLoop()
        # Plan, reflect, evaluate (E06). Injected as one composition so a
        # deployment can swap any of the three; deterministic by default,
        # so deliberating about a reply costs no completions.
        self._deliberation = deliberation or OutputQualityGate(
            evaluator=IndependentScorer(cascade) if cascade is not None else None
        )
        # Eight SLOs are configured; this is what actually observes them,
        # once per turn.
        self._accountant = TurnAccountant(slos=slos)
        self._logger = get_logger("customer_support_agent")
        self._graph = self._build()

    def adopt_conversation(self, previous: "CustomerSupportAgent") -> None:
        """Take over the conversation a previous build was holding.

        Who has verified, the order menu they were shown, the transcript
        and the observations a follow-up refers to all belong to the
        customer — not to the wiring. A rebuild swaps which controls are
        wired, and `/harness off` exists precisely to be asked the *same*
        question again; an agent that dropped the session would demand
        the customer code back before the comparison could be made.
        """
        self._sessions.update(previous._sessions)
        self._menus.update(previous._menus)
        self._conversations.update(previous._conversations)
        self._authoritative_history.update(previous._authoritative_history)
        self._last_supported_intent.update(previous._last_supported_intent)

    async def handle(
        self, query: str, session_id: str = "", customer: Customer | None = None
    ) -> AgentState:
        """Run one customer query end to end, on that customer's behalf.

        Prior turns are replayed so a follow-up has something to refer
        to. Without them "what about 10010" is unanswerable: the subject
        lives in the previous turn, not in the words the customer just
        typed.

        The customer is bound before any tool can run, and comes from
        authentication rather than from anything the model says. Passing
        one here binds it; passing none leaves whatever the caller already
        bound, so a web handler that authenticates upstream does not have
        to thread it through. With nothing bound either way, the tools
        scope to nothing — an unauthenticated request sees no orders
        rather than every order.
        """
        if customer is not None:
            # An explicitly supplied customer means the caller
            # authenticated upstream — an SSO session, a signed token, a
            # test fixture. The conversation does not re-ask for a code
            # it has already been given the answer to.
            if session_id:
                self._sessions[session_id] = customer
        # Bind identity *before* the graph runs, not inside a node.
        # Context variables copy into a task when it starts and do not
        # propagate back out, and LangGraph runs each node in its own
        # task — so a customer bound inside `authenticate` would be
        # invisible to `act`, and every lookup would silently scope to
        # nobody.
        customer = customer or self._sessions.get(session_id) or current_customer()
        set_current_customer(customer)
        self._control_loop = ToolControlLoop()
        initial = AgentState(
            query=query,
            session_id=session_id,
            actor=self._actor,
            customer_id=customer.customer_id if customer else None,
        )
        final = await self._graph.ainvoke(initial)
        state = AgentState.model_validate(final)
        self._remember_turn(session_id, query, state.response)
        if session_id and state.authoritative_tool_results:
            observations = self._authoritative_history.setdefault(session_id, [])
            observations.extend(state.authoritative_tool_results)
            max_observations = max(4, self._history_turns * 4)
            self._authoritative_history[session_id] = observations[-max_observations:]
        self._accountant.settle(self._score(state))
        return state

    def _score(self, state: AgentState) -> TurnOutcome:
        """Read the finished turn as an outcome the SLOs can observe.

        Everything here is derived from the state the graph already
        produced, rather than counted as it went. Counters threaded
        through eight nodes get updated in seven of them; a single read
        at the end cannot drift from what actually happened.
        """
        decisions = state.decisions
        return TurnOutcome(
            session_id=state.session_id,
            answered=bool(state.response) and not state.escalated,
            escalated=state.escalated,
            unauthorized_attempts=sum(1 for line in decisions if "not offered" in line),
            policy_refusals=sum(
                1 for line in decisions if "refused" in line and "not offered" not in line
            ),
            tool_calls=len(state.tool_results),
            tool_failures=sum(1 for line in decisions if ": failed" in line),
            retrievals=sum(1 for line in decisions if line.startswith("context:")),
            evaluator_calls=sum(1 for line in decisions if line.startswith("grounding:")),
            llm_calls=state.tool_rounds,
            usage=TokenUsage.model_validate(state.metadata.get("last_usage", {})),
            hallucination_rate=float(state.metadata.get("hallucination_rate", 0.0)),
            loop_aborted=state.tool_rounds >= self._max_rounds and not state.response,
            recoveries_attempted=int(state.metadata.get("recoveries_attempted", 0)),
            recoveries_succeeded=int(state.metadata.get("recoveries_succeeded", 0)),
        )

    @property
    def memory(self) -> MemoryFacade | None:
        """The memory this agent uses, for a console that wants to inspect it."""
        return self._memory

    def history_for(self, session_id: str) -> list[Message]:
        """The recent turns of one conversation, oldest first."""
        return list(self._conversations.get(session_id, []))

    def _remember_turn(self, session_id: str, query: str, response: str) -> None:
        """Keep the customer/agent exchange, bounded to the configured depth."""
        if not session_id or not response:
            return
        turns = self._conversations.setdefault(session_id, [])
        turns.append(Message(role="user", content=query))
        turns.append(Message(role="assistant", content=response))
        # Two messages per turn, so the cap is doubled.
        self._conversations[session_id] = turns[-self._history_turns * 2 :]
