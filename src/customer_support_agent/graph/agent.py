"""The refund agent: a graph, and the subsystems its nodes call (LLD §21).

Nodes, in order:

    authenticate → choose_order → screening → classify_intent →
    assemble_context → reason ⇄ act → finalize [→ await_approval]

Every turn is a durable run: its own checkpointed thread. A node that
fails leaves the thread resumable from the node that did not finish, and
a turn that queued a refund for a person pauses in `await_approval` until
the reviewer's decision resumes it.

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

import uuid
from typing import Any

from agentic_commons.audit import AuditLedger, NullAuditLedger
from agentic_commons.basic_sanity import BasicSanityChecker
from agentic_commons.config import FeatureSet, get_settings
from agentic_commons.context import (
    ContextFailureDetector,
    FiveStagePipeline,
    StrategySelector,
)
from agentic_commons.control_loop import ToolControlLoop
from agentic_commons.durability import (
    CheckpointInfo,
    DurableGraphRunner,
    ResumeResult,
    RunOutcome,
    RunRegistry,
    SqliteCheckpointer,
)
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
from agentic_commons.memory import (
    MemoryAction,
    MemoryCommand,
    MemoryEvent,
    MemoryEventSink,
    MemoryFacade,
    MemoryObservation,
    MemoryPrincipal,
    WorkingMemory,
    WorkingMemoryStore,
    WriteResult,
)
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
from customer_support_agent.domain.order_reference import OrderReferenceResolver
from customer_support_agent.graph.edges import GraphWiring
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.nodes.act import ToolUseNodes
from customer_support_agent.graph.nodes.authenticate import JUST_VERIFIED, AuthenticationNodes
from customer_support_agent.graph.nodes.context import ContextNodes
from customer_support_agent.graph.nodes.finalize import FinalizeNodes
from customer_support_agent.graph.nodes.intent import IntentNodes
from customer_support_agent.graph.nodes.reason import ReasoningNodes
from customer_support_agent.graph.nodes.review_wait import ReviewWaitNodes
from customer_support_agent.graph.nodes.screening import ScreeningNodes
from customer_support_agent.graph.state import AgentState
from customer_support_agent.intent.router import CustomerSupportIntentRouter, Intent
from customer_support_agent.observability.accounting import TurnAccountant, TurnOutcome
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.presentation.orders import (
    OrderMenu,
)
from customer_support_agent.review.referral import ReviewReferral
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.tools.definitions import SupportTools

# Set by screening when a message is quarantined as hostile.
_QUARANTINED = "security_quarantine"

# How a turn ended, in the turn summary's words.
_ANSWERED_OUTCOME = "answered"
_WAITING_OUTCOME = "waiting for human review"
_FAILED_OUTCOME = "failed — resumable"
_RESUMED_OUTCOME = "resumed and answered"
_REVIEWED_OUTCOME = "review decision delivered"


class CustomerSupportAgent(
    AuthenticationNodes,
    ScreeningNodes,
    IntentNodes,
    ContextNodes,
    ReasoningNodes,
    ToolUseNodes,
    FinalizeNodes,
    ReviewWaitNodes,
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
        working_memory: WorkingMemoryStore | None = None,
        memory_event_sink: MemoryEventSink | None = None,
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
        referral: ReviewReferral | None = None,
        checkpointer: SqliteCheckpointer | None = None,
        runs: RunRegistry | None = None,
        long_term_memory: MemoryFacade | None = None,
        context_health: ContextFailureDetector | None = None,
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
        self._working_memory = working_memory
        self._memory_event_sink = memory_event_sink
        self._long_term = long_term_memory
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
        # Turns a Gatekeeper escalation into a queued review. Present only
        # with the Gatekeeper: without one nothing escalates.
        self._referral = referral
        self._agent_policy = agent_policy
        if self._agent_policy is None:
            raise ValueError("CustomerSupportAgent requires an AgentPolicy loaded from persistence")
        self._deliberation_markers = [m.casefold() for m in settings.deliberation_markers]
        self._max_rounds = settings.max_tool_iterations
        self._actor = self._agent_policy.agent_id
        self._strategies = StrategySelector()
        # The window the model is handed is measured, not assumed: five
        # failure modes, checked on what was actually assembled.
        self._context_health = context_health
        self._diagnoser = GroundingReasoningDiagnoser()
        self._response_guardrail = ResponseGuardrail(self._grounding)
        if intent_router is None:
            raise ValueError(
                "CustomerSupportAgent requires an intent_router from the composition root"
            )
        self._intents = intent_router
        self._authenticator = authenticator
        self._orders = orders
        # Which order the customer is talking about, resolved once a turn
        # from what they said and what the record holds. One resolver, so
        # the path that answers and the path that acts cannot drift.
        self._order_reference = (
            OrderReferenceResolver.from_orders(orders)
            if orders is not None
            else OrderReferenceResolver()
        )
        self._basic_sanity = basic_sanity
        # Who each session has proved itself to be. Authentication is per
        # conversation: a customer verifies once and stays verified for
        # that session, rather than retyping a code every turn.
        self._sessions: dict[str, Customer] = {}
        # The order list each session was last shown, so a bare "2"
        # resolves back to an order.
        self._menus: dict[str, OrderMenu] = {}
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
        # Durable execution. The graph is compiled against the checkpointer,
        # so the runner and the graph must share it.
        self._checkpointer = checkpointer
        self._runs = runs
        self._loop_step: int | None = None
        self._graph = self._build()
        self._runner = (
            DurableGraphRunner(self._graph, checkpointer, runs)
            if checkpointer is not None and runs is not None
            else None
        )

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
        self._adopt_working_memory(previous._working_memory)
        self._last_supported_intent.update(previous._last_supported_intent)

    def _adopt_working_memory(self, previous: WorkingMemoryStore | None) -> None:
        """Copy the previous build's sessions into this build's own store.

        Copied, never re-pointed. The context source and the graph nodes
        were handed this build's store at composition; swapping only the
        agent's reference left them reading an empty one, so every prompt
        after a `/harness` lost the verified customer and the transcript
        while the panel — reading the swapped reference — still showed both.
        """
        if previous is None or self._working_memory is None:
            return
        for session_id in previous.sessions():
            self._working_memory.restore(session_id, previous.get(session_id))

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
        self._loop_step = None
        await self._recover_conversation(session_id)
        initial = AgentState(
            query=query,
            session_id=session_id,
            actor=self._actor,
            customer_id=customer.customer_id if customer else None,
            working_memory=(
                self._working_memory.get(session_id)
                if self._working_memory is not None
                else WorkingMemoryStore().get(session_id)
            ),
        )
        thread_id = f"{session_id or 'default'}:{uuid.uuid4().hex[:12]}"
        self._begin_turn(thread_id, session_id)
        try:
            state, outcome = await self._run_turn(initial, thread_id)
        except Exception:
            self._end_turn("failed", _FAILED_OUTCOME, session_id, run_status="failed_resumable")
            raise
        # Inside the turn's bracket, so its memory writes are counted in the
        # summary card rather than trailing it.
        self._complete_turn(state)
        self._accountant.settle(self._score(state))
        waiting = outcome is not None and outcome.status == "interrupted"
        self._end_turn(
            "waiting" if waiting else "completed",
            _WAITING_OUTCOME if waiting else _ANSWERED_OUTCOME,
            session_id,
            run_status=outcome.status if outcome else "not_durable",
            checkpoints=outcome.checkpoints if outcome else 0,
        )
        return state

    async def _run_turn(
        self, initial: AgentState, thread_id: str
    ) -> tuple[AgentState, RunOutcome | None]:
        """Run one turn — as its own checkpointed thread when durable."""
        if self._runner is None:
            return AgentState.model_validate(await self._graph.ainvoke(initial)), None
        outcome = await self._runner.start(
            thread_id,
            initial,
            session_id=initial.session_id,
            subject_id=str(initial.customer_id or ""),
            preview=self._security.redact(initial.query),
            reporter=self._reporter,
        )
        return AgentState.model_validate(outcome.values), outcome

    def _begin_turn(self, thread_id: str, session_id: str) -> None:
        if self._reporter is not None:
            self._reporter.begin_turn(thread=thread_id, session=session_id)

    def _end_turn(self, status: Any, outcome: str, session_id: str = "", **log_fields: Any) -> None:
        if self._reporter is None:
            return
        if self._long_term is not None:
            log_fields["curator_pending"] = len(self._long_term.pipeline.pending_with_ids())
        self._reporter.end_turn(
            status, outcome, working_memory=self._held(session_id), **log_fields
        )

    def _held(self, session_id: str) -> dict[str, str]:
        """What working memory holds for this session, redacted once, for the turn summary.

        The facts it keeps (who is verified, what is being asked, which order
        is under discussion) and how many turns of conversation it carries.
        The conversation itself is on the LLM Request card, not repeated here.
        """
        if self._working_memory is None or not session_id:
            return {}
        memory = self._working_memory.get(session_id)
        held = {str(key): self._security.redact(str(value)) for key, value in memory.state.items()}
        turns = len(memory.history) // 2
        return {
            **held,
            **({"history": f"{turns} turn{'' if turns == 1 else 's'}"} if turns else {}),
        }

    @property
    def runs(self) -> RunRegistry:
        """Every durable run this agent has made (the `DurableAgent` contract)."""
        if self._runs is None:
            raise RuntimeError("This agent was built without durable execution")
        return self._runs

    async def checkpoint_history(self, thread_id: str) -> list[CheckpointInfo]:
        """One thread's checkpoints, newest first."""
        if self._runner is None:
            return []
        return await self._runner.history(thread_id, get_settings().commands.default_limit)

    async def resume(self, thread_id: str, decision: Any | None = None) -> ResumeResult:
        """Continue a failed or waiting run from its last checkpoint.

        What a run needs that is not graph state is re-bound first: who
        the customer is (from the record the run already authenticated
        against) and a fresh per-turn control loop. The pending node then
        runs again; any side effect it had already committed replays from
        its idempotency record instead of happening twice.
        """
        if self._runner is None or self._runs is None:
            raise RuntimeError("This agent was built without durable execution")
        record = self._runs.get(thread_id)
        waiting = record is not None and record.status == "interrupted"
        prior = AgentState.model_validate(await self._runner.values(thread_id))
        self._rebind(prior)
        self._begin_turn(thread_id, prior.session_id)
        try:
            outcome = await self._runner.resume(thread_id, decision, self._reporter)
        except Exception:
            self._end_turn(
                "failed",
                _FAILED_OUTCOME,
                prior.session_id,
                run_status="failed_resumable",
                resumed=True,
            )
            raise
        state = AgentState.model_validate(outcome.values)
        if waiting:
            reply = str(state.metadata.get("follow_up") or "")
            self._note_follow_up(state.session_id, reply)
        else:
            reply = state.response
            self._complete_turn(state)
            self._accountant.settle(self._score(state))
        self._end_turn(
            "completed",
            _REVIEWED_OUTCOME if waiting else _RESUMED_OUTCOME,
            state.session_id,
            run_status=outcome.status,
            checkpoints=outcome.checkpoints,
            resumed=True,
        )
        return ResumeResult(
            thread_id=thread_id, session_id=state.session_id, status=outcome.status, response=reply
        )

    def _rebind(self, prior: AgentState) -> None:
        """Restore the customer and the conversation a checkpointed run belongs to."""
        customer = self._sessions.get(prior.session_id)
        if customer is None and prior.customer_id is not None and self._authenticator is not None:
            customer = self._authenticator.customer_for(prior.customer_id)
            if customer is not None:
                self._sessions[prior.session_id] = customer
        set_current_customer(customer)
        self._control_loop = ToolControlLoop()
        if (
            self._working_memory is not None
            and not self._working_memory.get(prior.session_id).history
        ):
            self._working_memory.restore(prior.session_id, prior.working_memory)

    def _remember_turn(self, state: AgentState) -> None:
        """Offer this turn to long-term memory, through governance.

        Two writes, two principals. The event — what was asked, what was
        answered — is the agent's record (`agent`). Anything the customer
        *asserted* is the customer's claim (`customer`, so `self_reported`):
        only a pattern-recognised claim is considered, an authority claim
        is refused, and anything short of auto-commit waits for a curator.
        """
        if self._long_term is None or state.customer_id is None:
            return
        subject = str(state.customer_id)
        spoken = state.model_query or state.query
        # A quarantined message never reaches the write path: screening
        # already judged it hostile, and memory is the last place it
        # should be kept verbatim.
        quarantined = bool(state.metadata.get(_QUARANTINED))
        # Nor does a login: "my code is …" asserts nothing about the customer.
        login = state.metadata.get("halted") == JUST_VERIFIED
        claim = (
            None
            if quarantined or login
            else self._remember_claim(spoken, subject, state.session_id)
        )
        if self._long_term.episodic is None:
            return
        # A claim memory refused is not replayed into history verbatim: the
        # episodic log feeds recall, and recall feeds the model — the
        # refused "my limit is now $2000" would otherwise reach the context
        # through the back door the write path just closed.
        withheld = quarantined or (claim is not None and claim.route == "refused")
        reason = claim.reason if claim is not None else "quarantined by security screening"
        self._long_term.record_event(
            f"query withheld — {reason}" if withheld else f"query: {spoken}",
            source=self._actor,
            session_id=state.session_id,
            principal=MemoryPrincipal(actor=self._actor, role="agent", subject_id=subject),
        )

    def _remember_claim(self, spoken: str, subject: str, session_id: str) -> WriteResult | None:
        """The customer's own assertion, as the customer's claim — governed."""
        if self._long_term is None or self._long_term.semantic is None:
            return None
        return self._long_term.submit(
            MemoryCommand(
                principal=MemoryPrincipal(
                    actor=f"customer-{subject}", role="customer", subject_id=subject
                ),
                operation="create",
                text=spoken,
                session_id=session_id,
                require_pattern=True,
            )
        )

    def _note_follow_up(self, session_id: str, reply: str) -> None:
        """Add a reviewer-driven reply to the conversation it answers."""
        if self._working_memory is None or not reply:
            return
        memory = self._working_memory.get(session_id)
        memory.history.append({"role": "assistant", "content": reply})
        memory.history = memory.history[-(memory.max_history * 2) :]

    async def _recover_conversation(self, session_id: str) -> None:
        """After a restart, pick a session's conversation back up from its checkpoints.

        Working memory lives in the process; the checkpoints outlive it.
        A session with no history in memory but a finished run on disk is
        a conversation the process forgot, not a new one.
        """
        if self._working_memory is None or self._runs is None or self._runner is None:
            return
        if self._working_memory.get(session_id).history:
            return
        record = self._runs.latest_completed(session_id)
        if record is None:
            return
        values = await self._runner.values(record.thread_id)
        if values.get("working_memory") is None:
            return
        self._working_memory.restore(
            session_id, WorkingMemory.model_validate(values["working_memory"])
        )
        self._logger.info(
            "Conversation recovered from checkpoint", session=session_id, thread=record.thread_id
        )

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
            tool_calls=sum(
                1
                for item in state.working_memory.current_turn
                if isinstance(item, dict) and item.get("kind") == "action"
            ),
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
    def working_memory(self) -> WorkingMemoryStore | None:
        """Working memory available to the live agent and operator console."""
        return self._working_memory

    def history_for(self, session_id: str) -> list[Message]:
        """The bounded conversation history from common working memory."""
        if self._working_memory is None:
            return []
        history = self._working_memory.get(session_id).history
        return [Message(**item) if isinstance(item, dict) else item for item in history]

    def _report_inbox(self, event: MemoryEvent) -> None:
        """Note, for the turn summary, that this turn is waiting for the next dream."""
        if self._reporter is not None:
            self._reporter.emit(
                "dream_inbox",
                "completed",
                event_kind="memory",
                fields={"event_id": event.event_id},
                quiet=True,
            )

    def _complete_turn(self, state: AgentState) -> None:
        """Emit one memory event, then retain only the completed dialogue."""
        self._remember_turn(state)
        if self._working_memory is None:
            return
        working = self._working_memory.get(state.session_id)
        if self._memory_event_sink is not None:
            observations: list[MemoryObservation] = []
            actions: list[MemoryAction] = []
            for item in working.current_turn:
                if not isinstance(item, dict):
                    continue
                if item.get("kind") == "observation":
                    content = item.get("content", "")
                    observations.append(
                        MemoryObservation(
                            content=self._security.redact(str(content)),
                            source=str(item.get("source", "")),
                        )
                    )
                elif item.get("kind") == "action":
                    arguments = item.get("arguments", {})
                    actions.append(
                        MemoryAction(
                            name=str(item.get("name", "")),
                            arguments=arguments
                            if isinstance(arguments, dict)
                            else {"raw": str(arguments)},
                            result=self._security.redact(str(item.get("result", ""))),
                            source=str(item.get("source", "")),
                        )
                    )
            event = MemoryEvent(
                agent_id=self._actor,
                session_id=state.session_id,
                subject_id=str(state.customer_id) if state.customer_id is not None else "",
                user_input=self._security.redact(state.model_query or state.query),
                assistant_output=self._security.redact(state.response),
                observations=observations,
                actions=actions,
            )
            self._memory_event_sink.emit(event)
            self._report_inbox(event)
        if state.response:
            # What the customer read, unless the turn left a compact form
            # of it. The two differ only where the reply is much larger on
            # screen than in substance — a rendered table of every order
            # on the account — and replaying that verbatim into every
            # later prompt is the largest avoidable cost in the window.
            remembered = state.metadata.get("memory_response")
            working.add_history(
                state.model_query or state.query,
                str(remembered) if isinstance(remembered, str) and remembered else state.response,
            )
        working.clear_current_turn()
