"""The LangGraph refund agent (LLD §21, HLD §4 request lifecycle).

Nodes, in order:

    authenticate → choose_order → basic_sanity → intake →
    assemble_context → reason ⇄ act → finalize

Every node calls into `agentic_commons` and owns no platform logic of
its own. That is the invariant this file exists to hold: the running
product is a *composition*, and a second LLM client or a second auth
check appearing here would be the duplication regression that was found
and fixed once already.

The dependencies arrive through `build_agent(...)`. Nothing is
constructed inline, so a test can substitute a scripted LLM and an
in-memory database without patching a single import.
"""

import json
import re
from collections import Counter
from typing import Any

from langgraph.graph import END, StateGraph

from agentic_commons.audit import AuditEntry, AuditLedger, NullAuditLedger
from agentic_commons.basic_sanity import BasicSanityChecker, BasicSanityRequest
from agentic_commons.config import FeatureSet, get_settings
from agentic_commons.context import (
    Chunk,
    ContextFailureDetector,
    FiveStagePipeline,
    StrategySelector,
    Task,
)
from agentic_commons.control_loop import ControlLoopReport, ToolControlLoop
from agentic_commons.evals import NollmScoringCascadeImpl
from agentic_commons.executor import (
    ApprovedToolCall,
    ExecutionOutcome,
    ToolExecutionHandler,
    ToolExecutor,
)
from agentic_commons.gatekeeper import AuthorityContext, Gatekeeper
from agentic_commons.guardrails import CustomerResponse, ResponseGuardrail
from agentic_commons.hallucination_protection import DetectionLadder, GroundingReasoningDiagnoser
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.embeddings import tokenize
from agentic_commons.llm.types import CompletionRequest, CompletionResult, Message, TokenUsage
from agentic_commons.logging import get_logger
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability import (
    SLORegistry,
    StageReporter,
    Tracer,
)
from agentic_commons.observability.types import utc_now
from agentic_commons.resilience import (
    DeadLetterQueue,
    ResilientCaller,
    external_api_error_fields,
)
from agentic_commons.security import SecuritySuite
from agentic_commons.security.auth.policy import AgentPolicy
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agentic_commons.tools.types import ToolResult
from customer_support_agent.config import get_agent_settings
from customer_support_agent.control_loop import CascadeEvaluator, DeliberationLoop
from customer_support_agent.core.accounting import TurnAccountant, TurnOutcome
from customer_support_agent.core.authentication import CustomerAuthenticator
from customer_support_agent.core.identity import (
    Customer,
    current_customer,
    set_current_customer,
)
from customer_support_agent.core.intent import CustomerSupportIntentRouter, Intent
from customer_support_agent.core.presentation import (
    OrderMenu,
    build_menu,
    customer_orders,
    render_order_detail,
    render_order_list,
    resolve_selection,
)
from customer_support_agent.core.repositories import OrdersRepository
from customer_support_agent.core.schema import AgentState, policy_text
from customer_support_agent.core.tools import SupportTools
from customer_support_agent.prompts.system_prompt import (
    build_reply_instruction,
    build_system_prompt,
)

# Shown when the model produces nothing a customer could read.
# Keep it free of unsupported operational promises.
_FALLBACK_REPLY = (
    "Sorry — I wasn't able to complete that. Please try again."
)


# Each optional stage and the feature that decides whether it is built.
_STAGE_FEATURES = {
    "basic_sanity": "basic_sanity",
    "prompt_injection": "prompt_injection",
    "content_moderation": "content_moderation",
    "classify_intent": "intent_classification",
    "assemble_context": "context",
}


class CustomerSupportAgent:
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
        deliberation: DeliberationLoop | None = None,
        grounding: DetectionLadder | None = None,
        authenticator: CustomerAuthenticator | None = None,
        orders: OrdersRepository | None = None,
        basic_sanity: BasicSanityChecker | None = None,
        agent_policy: AgentPolicy | None = None,
        intent_router: CustomerSupportIntentRouter | None = None,
        features: FeatureSet | None = None,
        handler: ToolExecutionHandler | None = None,
        audit: AuditLedger | None = None,
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
        self._deliberation = deliberation or DeliberationLoop(
            evaluator=CascadeEvaluator(cascade) if cascade is not None else None
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

    # ---- nodes -------------------------------------------------------

    async def authenticate(self, state: AgentState) -> dict[str, Any]:
        """Establish who is speaking before looking anything up.

        Until a valid code arrives there is no owner to scope a lookup
        to, so "where is my order" has no answer. Verification is a
        database lookup, not a model call: the code either matches a row
        or it does not, and a deterministic check cannot be talked into a
        different answer.
        """
        existing = self._sessions.get(state.session_id) or current_customer()
        self._report(
            "authentication",
            summary="Verifying the customer",
        )
        if existing is not None:
            # Already verified — either earlier in this conversation, or
            # by the caller before it handed the request over.
            set_current_customer(existing)
            if state.session_id:
                self._sessions[state.session_id] = existing
            self._done(
                "authentication",
                summary="Customer already verified",
                fields={
                    "verified": True,
                    "method": "session",
                    "customer_id": existing.customer_id,
                },
            )
            return {"customer_id": existing.customer_id}
        if self._authenticator is None:
            self._done(
                "authentication",
                summary="Authentication delegated upstream",
                fields={"verified": False, "method": "upstream"},
            )
            return {}
        result = self._authenticator.authenticate(state.query)
        if not result.authenticated:
            if result.code_attempted:
                self._reporter_blocked(
                    "authentication",
                    summary="Customer code was not accepted",
                    fields={
                        "verified": False,
                        "method": "customer_code",
                        "reason": "invalid_or_unknown_code",
                    },
                )
            else:
                self._reporter_waiting(
                    "authentication",
                    summary="Waiting for customer verification",
                    fields={
                        "verified": False,
                        "method": "customer_code",
                        "action": "request_code",
                    },
                )
            return {
                "response": result.message,
                "decisions": [
                    *state.decisions,
                    "auth: rejected" if result.code_attempted else "auth: awaiting code",
                ],
                "metadata": {**state.metadata, "halted": "unauthenticated"},
            }
        assert result.customer is not None
        if state.session_id:
            self._sessions[state.session_id] = result.customer
            if result.menu is not None:
                self._menus[state.session_id] = result.menu
        set_current_customer(result.customer)
        self._done(
            "authentication",
            summary="Customer verified",
            fields={
                "verified": True,
                "method": "customer_code",
                "customer_id": result.customer.customer_id,
                "orders": result.menu.size if result.menu is not None else 0,
            },
        )
        self._note(
            state,
            "verified_customer",
            f"{result.customer.name} (account #{result.customer.customer_id})",
            category="identity",
        )
        return {
            "response": result.message,
            "customer_id": result.customer.customer_id,
            "decisions": [*state.decisions, f"auth: verified #{result.customer.customer_id}"],
            # The verified read-back *is* the answer to "here is my code".
            # Nothing further is needed this turn.
            "metadata": {**state.metadata, "halted": "just_verified"},
        }

    async def choose_order(self, state: AgentState) -> dict[str, Any]:
        """Answer a menu choice straight from the database.

        "2" after being shown a list is a lookup, not a conversation.
        Resolving it here costs one SELECT; routing it through the model
        costs a completion to read a row aloud, and risks the model
        getting a digit wrong about the customer's own order.

        Only an explicit menu-number reply lands here.
        "order 10560" is an order reference, not a menu command, and
        continues through the normal conversational path.
        """
        menu = self._menus.get(state.session_id)
        if menu is None or self._orders is None or state.customer_id is None:
            return {}
        if _asks_for_the_list(state.query):
            orders = customer_orders(self._orders, state.customer_id)
            self._menus[state.session_id] = build_menu(orders)
            self._report(
                "order_lookup",
                summary="Loading the customer's orders",
            )
            self._done(
                "order_lookup",
                summary="Order list loaded",
                fields={"count": len(orders)},
            )
            return self._halt(state, render_order_list(orders), "menu: listed")
        order_id = resolve_selection(state.query, menu)
        if order_id is None:
            return {}
        self._report(
            "order_lookup",
            summary="Checking order details",
            fields={"order_id": order_id},
        )
        order = self._orders.get_for_customer(order_id, state.customer_id)
        if order is None:
            self._reporter_blocked(
                "order_lookup",
                summary="Order not found on this account",
                fields={"order_id": order_id, "found": False},
            )
            return self._halt(
                state, f"I couldn't find order {order_id} on your account.", "menu: not found"
            )
        self._done(
            "order_lookup",
            summary="Order found",
            fields={
                "order_id": order.order_id,
                "status": order.order_status,
                "refund_status": order.refund_status,
                "amount": f"${order.amount:,.2f}",
                "found": True,
            },
        )
        self._note(
            state,
            "order_under_discussion",
            f"order {order.order_id}, {order.order_status}, ${order.amount:,.2f}",
            category="order",
        )
        return self._halt(state, render_order_detail(order), f"menu: order {order_id}")

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

    async def basic_sanity(self, state: AgentState) -> dict[str, Any]:
        """Run cheap, deterministic domain checks before any model call.

        Reached only when this agent was given a checker — `_build` omits
        the node otherwise, so there is no "not configured" step to
        narrate. The checks are an *early* duplicate, not the only
        enforcement: business rules, the tools and the review queue all
        re-ask the same questions. Withholding this stage therefore costs
        a model call and a cheap refusal, it does not open a hole.
        """
        assert self._basic_sanity is not None
        self._report(
            "basic_sanity",
            event_kind="stage",
            summary="Running deterministic pre-LLM sanity checks",
        )
        result = self._basic_sanity.check(
            BasicSanityRequest(
                query=state.query,
                agent_id=self._actor,
                session_id=state.session_id,
                actor=state.actor,
                subject_id="" if state.customer_id is None else str(state.customer_id),
                metadata=state.metadata,
            )
        )
        fields = _basic_sanity_fields(result)

        if result.passed:
            deferred = result.reason_code == "ORDER_REFERENCE_NOT_PRESENT"
            order_id = result.facts.get("order_id") if isinstance(result.facts, dict) else None
            if isinstance(order_id, int):
                # This is the authoritative current order reference for the
                # conversation. Store only the identifier in working memory;
                # order data itself must still come from an ownership-scoped
                # repository/tool read on the path that needs it.
                self._note(
                    state,
                    "order_under_discussion",
                    f"order {order_id}",
                    category="order",
                )
            self._done(
                "basic_sanity",
                summary=(
                    "Order reference not provided · validation deferred"
                    if deferred
                    else "Basic sanity checks passed"
                ),
                fields=fields,
            )
            metadata = {
                **state.metadata,
                "basic_sanity": result.model_dump(mode="json"),
            }
            if isinstance(order_id, int):
                metadata["order_under_discussion"] = order_id
            return {
                "decisions": [
                    *state.decisions,
                    (
                        f"basic_sanity: deferred ({result.reason_code})"
                        if deferred
                        else f"basic_sanity: pass ({result.reason_code})"
                    ),
                ],
                "metadata": metadata,
            }

        self._reporter_blocked(
            "basic_sanity",
            summary=f"Basic sanity blocked: {result.reason_code}",
            fields=fields,
        )
        return {
            "response": result.message
            or "I couldn't safely continue with that request.",
            "decisions": [
                *state.decisions,
                f"basic_sanity: blocked ({result.reason_code})",
            ],
            "metadata": {
                **state.metadata,
                "basic_sanity": result.model_dump(mode="json"),
                "halted": "basic_sanity",
            },
        }

    def _quarantine_security_request(
        self,
        state: AgentState,
        *,
        reason: str,
        injection_detected: bool = False,
        injection_reason: str = "",
        moderation_flagged: bool = False,
        moderation_reason: str = "",
        moderation_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persist a security-rejected request and stop before intent/LLM execution."""
        entry = None
        enqueue_error = None
        if self._dead_letter_queue is not None and self._dead_letter_queue.enabled:
            try:
                sanitized_query = self._security.redact(state.query)
                entry = self._dead_letter_queue.enqueue(
                    reason=reason,
                    payload=sanitized_query,
                    trace_id=str(state.metadata.get("trace_id", "")),
                    session_id=state.session_id,
                    customer_id=state.customer_id,
                    metadata={
                        "prompt_injection": injection_detected,
                        "content_moderation": moderation_flagged,
                        "injection_reason": injection_reason,
                        "moderation_categories": list(moderation_categories or []),
                        "moderation_reason": moderation_reason,
                    },
                )
            except Exception as exc:
                # Security remains fail-closed even if the quarantine store is
                # temporarily unavailable. Never continue to intent or the LLM.
                enqueue_error = str(exc)
                self._logger.error(
                    "Security quarantine store failed; request remains blocked",
                    error=enqueue_error,
                )

        self._done(
            "dead_letter_queue",
            summary="Request quarantined for human review",
            event_kind="escalation",
            fields={
                "queued": entry is not None,
                "queue_id": entry.id if entry is not None else None,
                "reason": reason,
                "prompt_injection": injection_detected,
                "content_moderation": moderation_flagged,
                "persistence_error": bool(enqueue_error),
            },
        )
        return {
            "response": "I’m unable to help with that request right now. Please rephrase your request.",
            "decisions": [
                *state.decisions,
                "security: request quarantined",
            ],
            "metadata": {
                **state.metadata,
                "halted": "security_quarantine",
                "security_quarantine": True,
                "dead_letter_queue_id": entry.id if entry is not None else None,
            },
        }

    async def prompt_injection(self, state: AgentState) -> dict[str, Any]:
        """Screen untrusted input before any of it reaches a prompt.

        A node rather than a step inside one, so that an agent configured
        without this defence has no injection stage in its graph at all —
        the trace shows the gap instead of showing a check that passed
        without looking at anything.
        """
        self._report(
            "prompt_injection_check",
            event_kind="security",
            summary="Checking untrusted input for injection",
        )
        assessment = await self._security.prepare_untrusted_async(
            self._system_prompt(), state.query, self._registry.all_tools()
        )
        injection_checks = list(getattr(assessment, "checks", []))
        jailbreak_check = next(
            (check for check in injection_checks if check.get("name") == "NVIDIA NemoGuard JailbreakDetect"),
            None,
        )
        deterministic_injection = any(
            check.get("name") == "Deterministic prompt-injection scan"
            and check.get("status") == "FAIL"
            for check in injection_checks
        )
        injection_fields: dict[str, Any] = {
            "deterministic": "MATCH" if deterministic_injection else "NO_MATCH"
        }
        if deterministic_injection:
            injection_fields["reason"] = assessment.reason
            self._reporter_warning(
                "prompt_injection_check",
                summary="Prompt injection blocked · deterministic match",
                event_kind="security",
                fields=injection_fields,
            )
        elif jailbreak_check and jailbreak_check.get("status") == "FAIL":
            injection_fields.update({
                "model": jailbreak_check.get("model", "unknown"),
                "score": jailbreak_check.get("score"),
                "latency_ms": jailbreak_check.get("latency_ms"),
                "reason": jailbreak_check.get("reason", "Hosted jailbreak guard flagged the input"),
            })
            self._reporter_warning(
                "prompt_injection_check",
                summary="Prompt injection blocked · model detection",
                event_kind="security",
                fields=injection_fields,
            )
        elif jailbreak_check and jailbreak_check.get("status") == "SKIPPED":
            injection_fields.update({
                "model": jailbreak_check.get("model", "unknown"),
                "latency_ms": jailbreak_check.get("latency_ms"),
                "error_code": jailbreak_check.get("error_code"),
                "http_status": jailbreak_check.get("http_status"),
                "reason": jailbreak_check.get("reason", "Hosted jailbreak guard unavailable"),
            })
            self._reporter_warning(
                "prompt_injection_check",
                summary="Prompt injection model check unavailable",
                event_kind="security",
                fields=injection_fields,
            )
        elif jailbreak_check:
            injection_fields.update({
                "model": jailbreak_check.get("model", "unknown"),
                "score": jailbreak_check.get("score"),
                "latency_ms": jailbreak_check.get("latency_ms"),
            })
            self._done(
                "prompt_injection_check",
                summary="Prompt injection check passed · model verified",
                event_kind="security",
                fields=injection_fields,
            )
        else:
            self._done(
                "prompt_injection_check",
                summary="Prompt injection check passed · deterministic scan",
                event_kind="security",
                fields=injection_fields,
            )

        if assessment.suspicious:
            return self._quarantine_security_request(
                state,
                reason="prompt_injection",
                injection_detected=True,
                injection_reason=assessment.reason,
            )
        return {
            "decisions": [*state.decisions, f"injection: {assessment.reason}"],
            "metadata": {
                **state.metadata,
                "allowed_actions": assessment.allowed_actions,
            },
        }

    async def content_moderation(self, state: AgentState) -> dict[str, Any]:
        """Screen the message against the content policy."""
        self._report(
            "content_moderation",
            event_kind="security",
            summary="Checking content policy",
        )
        moderation = await self._security.moderate_async(state.query)
        moderation_checks = list(getattr(moderation, "checks", []))
        content_guard_check = next(
            (check for check in moderation_checks if check.get("name") == "NVIDIA Nemotron Safety Guard 8B v3"),
            None,
        )
        deterministic_content = bool(moderation.matched_terms)
        moderation_fields: dict[str, Any] = {
            "deterministic": "MATCH" if deterministic_content else "NO_MATCH"
        }
        if deterministic_content:
            moderation_fields["reason"] = moderation.reason
            self._reporter_warning(
                "content_moderation",
                summary="Content safety blocked · deterministic match",
                event_kind="security",
                fields=moderation_fields,
            )
        elif content_guard_check and content_guard_check.get("status") == "FAIL":
            moderation_fields.update({
                "model": content_guard_check.get("model", "unknown"),
                "categories": list(content_guard_check.get("categories", [])),
                "latency_ms": content_guard_check.get("latency_ms"),
                "reason": content_guard_check.get("reason", "Hosted content-safety guard flagged the input"),
            })
            self._reporter_warning(
                "content_moderation",
                summary="Content safety blocked · model detection",
                event_kind="security",
                fields=moderation_fields,
            )
        elif content_guard_check and content_guard_check.get("status") == "SKIPPED":
            moderation_fields.update({
                "model": content_guard_check.get("model", "unknown"),
                "latency_ms": content_guard_check.get("latency_ms"),
                "error_code": content_guard_check.get("error_code"),
                "http_status": content_guard_check.get("http_status"),
                "reason": content_guard_check.get("reason", "Hosted content-safety guard unavailable"),
            })
            self._reporter_warning(
                "content_moderation",
                summary="Content safety model check unavailable",
                event_kind="security",
                fields=moderation_fields,
            )
        elif content_guard_check:
            moderation_fields.update({
                "model": content_guard_check.get("model", "unknown"),
                "latency_ms": content_guard_check.get("latency_ms"),
            })
            self._done(
                "content_moderation",
                summary="Content safety check passed · model verified",
                event_kind="security",
                fields=moderation_fields,
            )
        else:
            self._done(
                "content_moderation",
                summary="Content safety check passed · deterministic scan",
                event_kind="security",
                fields=moderation_fields,
            )

        if moderation.flagged:
            return self._quarantine_security_request(
                state,
                reason="content_moderation",
                moderation_flagged=True,
                moderation_reason=moderation.reason,
                moderation_categories=list(moderation.categories),
            )
        return {
            "decisions": [*state.decisions],
            "metadata": {**state.metadata},
        }

    async def classify_intent(self, state: AgentState) -> dict[str, Any]:
        """Decide what the customer is asking for, and what that permits.

        The Customer Support intent router is the only scope/intent
        mechanism. Its deterministic rules may reject only explicit,
        agent-specific out-of-scope requests. Everything else is sent to
        the BERT-family classifier; unresolved classifications remain
        UNKNOWN and are handed to the LLM with conversation context
        instead of being treated as out-of-scope.
        """
        self._report(
            "intent_classification",
            event_kind="intent",
            summary=(
                f"Loading intent classifier: {self._intents.model_name}"
                if not self._intents.model_ready
                else "Classifying the customer's request"
            ),
        )
        intent = await self._intents.classify_async(
            state.query, previous=self._last_supported_intent.get(state.session_id)
        )
        self._logger.info(
            "Customer Support intent resolved",
            **self._intent_status_fields(intent),
        )

        if intent.disposition.value == "out_of_scope":
            if state.session_id:
                # Explicitly out-of-scope turns clear the contextual business
                # intent; UNKNOWN/clarify turns must not poison a good prior one.
                self._last_supported_intent.pop(state.session_id, None)
            self._reporter_blocked(
                "intent_classification",
                summary="Intent rejected · deterministic rule",
                event_kind="intent",
                fields=self._intent_status_fields(intent),
            )
            return {
                "response": (
                    "I can help with your orders and refunds, "
                    "but I can't help with that request."
                ),
                "decisions": [
                    *state.decisions,
                    f"intent: out_of_scope ({intent.reason})",
                ],
                "metadata": {
                    **state.metadata,
                    "intent": intent.name,
                    "intent_confidence": intent.confidence,
                    "intent_disposition": intent.disposition.value,
                    "intent_source": intent.source,
                    "intent_model": intent.model,
                    "intent_score_type": intent.score_type,
                    "intent_candidates": [c.model_dump(mode="json") for c in intent.candidates],
                    "intent_reason": intent.reason,
                    "halted": "out_of_scope",
                },
            }

        if intent.disposition.value == "clarify":
            # A lightweight classifier is a routing optimization, not the
            # final semantic authority. Preserve the ambiguity and let the
            # LLM resolve it from conversation context. No business
            # capability is granted by this branch; Gatekeeper remains the
            # authority if the model proposes a tool.
            self._reporter_waiting(
                "intent_classification",
                summary="Intent unresolved · handing to contextual LLM",
                event_kind="intent",
                fields={
                    **self._intent_status_fields(intent),
                    "fallback": "llm_context",
                },
            )
            intent_route = "llm_fallback"
        else:
            intent_route = None

        if state.session_id and intent.name != "UNKNOWN":
            self._last_supported_intent[state.session_id] = intent
        plan = self._deliberation.plan(state.query, intent.name, list(intent.allowed_tools))
        intent_route = intent_route or self._intent_route(intent)
        if intent.name == "UNKNOWN":
            intent_route = "llm_fallback"
        self._done(
            "intent_classification",
            summary=self._intent_status_summary(intent),
            event_kind="intent",
            fields=self._intent_status_fields(intent),
        )
        self._note(state, "current_intent", intent.name, category="intent")
        decisions = [f"intent: {self._intent_decision_label(intent)}"]
        if self._memory is not None:
            self._memory.record_event(f"query: {state.query}", self._actor, state.session_id)
            decisions.extend(self._offer_to_memory(state))
        if not plan.is_single_step:
            decisions.append(f"plan: {len(plan.steps)} steps — {plan.steps[0].description}")
        return {
            "decisions": [*state.decisions, *decisions],
            "metadata": {
                **state.metadata,
                "intent": intent.name,
                "intent_confidence": intent.confidence,
                "intent_disposition": intent.disposition.value,
                "intent_source": intent.source,
                "intent_model": intent.model,
                "intent_score_type": intent.score_type,
                "intent_candidates": [c.model_dump(mode="json") for c in intent.candidates],
                "intent_tools": intent.allowed_tools,
                "intent_execution_route": intent_route,
                "plan": [step.model_dump() for step in plan.steps],
            },
        }

    async def read_intent(self, state: AgentState) -> dict[str, Any]:
        """Execute a classified read capability without an LLM completion.

        Intent selects a read-only capability; this node then checks that the
        authenticated customer owns the referenced order before invoking the
        read tool. The Gatekeeper still performs its trusted agent/tool
        authorization checks inside `_invoke`.
        """
        tool = self._read_tool_for_state(state)
        if tool is None:
            return {}

        order_id = self._resolve_read_order_id(state)
        if order_id is None:
            self._reporter_waiting(
                "read_authorization",
                summary="Read capability needs an order reference",
                event_kind="authorization",
                fields={"intent": state.metadata.get("intent", ""), "tool": tool.qualified_name},
            )
            return self._halt(
                state,
                "Which order should I check? Please provide the order number.",
                "read: awaiting order reference",
            )

        self._report(
            "read_authorization",
            event_kind="authorization",
            summary="Checking customer access to the requested order",
            fields={"order_id": order_id, "intent": state.metadata.get("intent", "")},
        )
        owned = self._orders.get_for_customer(order_id, state.customer_id) if self._orders else None
        if owned is None:
            fields = {
                "order_id": order_id,
                "customer_id": state.customer_id,
                "allowed": False,
                "reason_code": "ORDER_NOT_OWNED_BY_CUSTOMER",
            }
            self._reporter_blocked(
                "read_authorization",
                summary="Read access denied for this order",
                event_kind="authorization",
                fields=fields,
            )
            return self._halt(
                state,
                f"I couldn't find order {order_id} on your account.",
                f"read: access denied for order {order_id}",
            )

        self._done(
            "read_authorization",
            event_kind="authorization",
            summary="Customer is authorized to read this order",
            fields={
                "order_id": order_id,
                "customer_id": state.customer_id,
                "allowed": True,
                "permission": tool.required_permission or "none",
            },
        )

        # Absent means the injection stage was never built, so nothing
        # narrowed the surface. An empty *list* is a real narrowing to
        # nothing and still blocks — the two must not be conflated, or
        # switching the guard off would block every read instead.
        allowed_actions = state.metadata.get("allowed_actions")
        if isinstance(allowed_actions, list) and tool.qualified_name not in set(allowed_actions):
            self._reporter_blocked(
                "read_authorization",
                summary="Read capability blocked by security tool scope",
                event_kind="authorization",
                fields={
                    "order_id": order_id,
                    "tool": tool.qualified_name,
                    "reason_code": "READ_TOOL_NOT_ALLOWED",
                },
            )
            return self._halt(
                state,
                "I couldn't safely access that order with the permissions available for this request.",
                f"read: security scope denied {tool.name}",
            )

        payload = json.dumps({"order_id": order_id})
        text, decision, _stage, invocation = await self._invoke(
            tool.qualified_name,
            payload,
            state,
            {tool.qualified_name},
        )
        self._note(
            state,
            f"result_of_{tool.name}",
            text,
            category="tool_result",
        )
        base = self._halt(
            state,
            text,
            f"read: {tool.name} for order {order_id}"
            if invocation.get("tool_status") == "completed"
            else decision,
        )
        authoritative = (
            [text]
            if invocation.get("tool_status") == "completed"
            else []
        )
        return {
            **base,
            "tool_results": [text],
            "authoritative_tool_results": authoritative,
            "metadata": {
                **base["metadata"],
                "last_tool_invocations": [invocation],
            },
        }

    def _read_tool_for_state(self, state: AgentState) -> Any | None:
        """Return the sole non-side-effecting tool granted by the intent."""
        names = list(state.metadata.get("intent_tools", []))
        tools = [tool for name in names if (tool := self._registry.get(name)) is not None]
        if len(tools) != 1 or tools[0].side_effecting:
            return None
        return tools[0]

    def _intent_route(self, intent: Intent) -> str:
        """Map intent to a graph capability route from trusted tool metadata."""
        if intent.name == "UNKNOWN":
            return "llm_fallback"
        tools = [tool for name in intent.allowed_tools if (tool := self._registry.get(name)) is not None]
        if len(tools) == 1 and not tools[0].side_effecting:
            return "deterministic_read"
        if any(tool.side_effecting for tool in tools):
            return "llm_write"
        return "llm"

    def _resolve_read_order_id(self, state: AgentState) -> int | None:
        """Resolve the most recent order under discussion, newest evidence first."""
        metadata_id = state.metadata.get("order_under_discussion")
        if isinstance(metadata_id, int) and metadata_id > 0:
            return metadata_id

        explicit = _extract_order_id(state.query)
        if explicit is not None:
            return explicit

        # Working memory is recency ordered. The first matching fact is the
        # most recent validated order reference; do not search older facts
        # once one is found. This keeps follow-ups anchored to the latest
        # order without asking the model to infer which identifier wins.
        if self._memory is not None and state.session_id:
            for fact in self._memory.active_window(state.session_id):
                if fact.key != "order_under_discussion":
                    continue
                match = re.search(r"\border\s+(\d+)\b", fact.value, flags=re.IGNORECASE)
                if match:
                    return int(match.group(1))
        return None

    @staticmethod
    def _intent_status_fields(intent: Intent) -> dict[str, Any]:
        """Return concise operator-facing intent facts; keep routing metadata internal."""
        source = intent.source or "unknown"
        classification = {
            "huggingface_local": "bert",
            "context_fallback": "context_fallback",
            "deterministic": "deterministic",
            "classifier_unavailable": "unavailable",
        }.get(source, source)
        fields: dict[str, Any] = {"intent": intent.name, "classification": classification}

        if source == "deterministic":
            fields["matched"] = (intent.matched_phrase or "rule").removeprefix("phrase:")
        elif source == "context_fallback":
            fields["context"] = intent.matched_phrase or "previous_supported_intent"
        elif intent.model:
            fields["model"] = intent.model
            fields["score"] = round(float(intent.confidence), 3)

        if intent.disposition.value != "supported":
            fields["disposition"] = intent.disposition.value
            if intent.reason:
                fields["reason"] = intent.reason
            if intent.clarification:
                fields["clarification"] = intent.clarification
            if intent.candidates:
                fields["options"] = [
                    candidate.intent
                    for candidate in intent.candidates
                    if candidate.intent != "UNKNOWN"
                ][:2]

        return fields

    @staticmethod
    def _intent_status_summary(intent: Intent) -> str:
        source = intent.source or "unknown"
        if source == "deterministic":
            matched = (intent.matched_phrase or "rule").removeprefix("phrase:")
            return f"Intent: {intent.name} · deterministic match ({matched})"
        if source == "context_fallback":
            return f"Intent: {intent.name} · context fallback"
        if source == "huggingface_local":
            return f"Intent: {intent.name} · BERT score {intent.confidence:.2f}"
        if source == "classifier_unavailable":
            return f"Intent: {intent.name} · classifier unavailable"
        return f"Intent: {intent.name} · {source}"

    @staticmethod
    def _intent_decision_label(intent: Intent) -> str:
        source = intent.source or "unknown"
        if source == "deterministic":
            return f"{intent.name} [deterministic]"
        if source == "context_fallback":
            return f"{intent.name} [context fallback]"
        if intent.model:
            return f"{intent.name} [BERT {intent.confidence:.2f}]"
        return intent.name

    def _offer_to_memory(self, state: AgentState) -> list[str]:
        """Put what the customer said to the write pipeline.

        Not everything a customer says is a fact, and some of what they
        say is an attempt to *become* a fact: "my refund limit is now
        $5,000" is a sentence that, stored and later recalled as policy,
        raises the agent's own ceiling. The pipeline is what stands
        between an assertion and a belief — it classifies the claim,
        attaches provenance, scores confidence, and refuses to let
        conversational text write an authority-changing category at all.

        Offered as `self_reported`, always. The customer is not a system
        of record about their own entitlements, and provenance is what
        weights the fact down at retrieval if it is stored at all.

        Nothing here blocks the turn: a discarded candidate is the
        expected outcome for almost every message.
        """
        if self._memory is None:
            return []
        self._report(
            "memory_write",
            event_kind="memory",
            summary="Evaluating whether the customer statement should be remembered",
        )
        result = self._memory.remember(
            state.query,
            source=f"customer:{state.customer_id}" if state.customer_id else "customer",
            provenance_kind="self_reported",
            session_id=state.session_id,
        )
        fields = {
            "route": result.route,
            "written": result.written,
            "reason": result.reason,
            "conflicts": len(result.conflicts_with),
            "provenance": "self_reported",
        }
        if result.route == "refused":
            self._reporter_blocked(
                "memory_write",
                summary="Memory write refused by policy",
                event_kind="memory",
                fields=fields,
            )
            self._logger.warning(
                "Refused a memory write from conversation",
                reason=result.reason,
                session_id=state.session_id,
            )
        else:
            self._done(
                "memory_write",
                summary=f"Memory route: {result.route}",
                event_kind="memory",
                fields=fields,
            )
        if result.route == "discard":
            return []
        return [f"memory: {result.route} ({result.reason})"]

    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window: policy, recalled memory, and the query."""
        self._report(
            "context_assembly",
            event_kind="context",
            summary="Assembling the model context window",
        )
        if self._context is None:
            context = policy_text()
            self._done(
                "context_assembly",
                summary="Policy context loaded",
                event_kind="context",
                fields={"source": "policy", "used_tokens": len(tokenize(context))},
            )
            return {"context": context}
        assembled = await self._context.assemble(
            Task(query=state.query, session_id=state.session_id, actor=state.actor)
        )
        fitted = self._strategies.fit(assembled)
        decisions = [f"context: {fitted.used_tokens}/{fitted.budget_tokens} tokens"]
        modes = self._context_health.assess(fitted, set(tokenize(state.query)))
        if modes:
            self._logger.warning("Context window shows failure modes", modes=modes)
            decisions.append(f"context health: {', '.join(modes)}")

        by_kind = Counter(chunk.kind for chunk in fitted.chunks)
        context_fields = {
            "chunks": len(fitted.chunks),
            "dropped": len(fitted.dropped),
            "budget_tokens": fitted.budget_tokens,
            "used_tokens": fitted.used_tokens,
            "utilization": (
                round(fitted.used_tokens / fitted.budget_tokens, 2)
                if fitted.budget_tokens
                else 0.0
            ),
            "kinds": dict(by_kind),
            "strategies": fitted.applied_strategies,
            "health": modes or ["healthy"],
        }
        if self._memory is not None and state.session_id:
            self._report(
                "memory_recall",
                event_kind="memory",
                summary="Checking working memory",
            )
            held = self._memory.window_size(state.session_id)
            self._done(
                "memory_recall",
                event_kind="memory",
                summary=f"Working memory: {held} fact(s)",
                fields={
                    "operation": "recall",
                    "scope": "working",
                    "facts_held": held,
                },
            )
        if modes:
            self._reporter_warning(
                "context_assembly",
                summary="Context health warning",
                event_kind="context",
                fields=context_fields,
            )
        else:
            self._done(
                "context_assembly",
                summary="Context window assembled",
                event_kind="context",
                fields=context_fields,
            )
        return {"context": fitted.text, "decisions": [*state.decisions, *decisions]}

    async def reason(self, state: AgentState) -> dict[str, Any]:
        """One model turn, with the tools this turn is allowed to use."""
        round_report = self._control_loop.evaluate_round(
            state.tool_rounds + 1, self._max_rounds
        )
        self._report(
            "llm_call",
            event_kind="model",
            summary="Selecting the next action",
        )
        tools = await self._offered_tools(state)
        offered = sorted(tool.qualified_name for tool in tools)
        request = CompletionRequest(
            messages=self._prompt_messages(state),
            tools=self._registry.to_openai_tools([tool.qualified_name for tool in tools]),
        )
        try:
            result = await self._call_model(request, state, f"{state.session_id}:{state.tool_rounds}")
        except Exception as exc:
            error_fields = external_api_error_fields(exc)
            self._reporter_warning(
                "llm_call",
                summary="LLM unavailable after retries",
                event_kind="resilience",
                fields={
                    "dependency": self._resilient.name,
                    **(error_fields or {}),
                    **({"reason": str(exc)} if not error_fields else {}),
                },
            )
            return {
                "messages": [Message(role="assistant", content=_FALLBACK_REPLY)],
                "tool_rounds": state.tool_rounds + 1,
                "escalated": True,
                "decisions": [*state.decisions, "resilience: llm unavailable after retries"],
                "metadata": {**state.metadata, "external_dependency_failed": True},
            }
        message = Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
        tool_names = [call.name for call in (result.tool_calls or [])]
        metrics = self._selector.last_metrics if self._selector is not None else None
        requested_tool_calls = [
            _safe_tool_request_fields(call.name, call.arguments)
            for call in (result.tool_calls or [])
        ]
        llm_fields = {
            "model": result.model,
            "finish_reason": result.finish_reason or "unknown",
            "tool_calls": len(result.tool_calls or []),
            "requested_tools": tool_names,
            "requested_tool_calls": requested_tool_calls,
            "tools_offered": offered,
            "input_tokens": result.usage.prompt_tokens,
            "output_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        }
        if metrics is not None:
            llm_fields.update(
                {
                    "tools_available": metrics.tools_available,
                    "tools_exposed": metrics.tools_exposed,
                    "tool_selection_latency_ms": round(metrics.latency_ms, 1),
                }
            )
        self._done(
            "llm_call",
            tokens=result.usage.total_tokens,
            event_kind="model",
            summary=(
                f"Requested tool: {tool_names[0]}"
                if tool_names
                else "Produced a final response"
            ),
            fields=llm_fields,
        )
        self._done(
            "control_loop",
            event_kind="control_loop",
            summary=(
                f"Round {state.tool_rounds + 1}: tool execution requested"
                if tool_names
                else f"Round {state.tool_rounds + 1}: ready to finalize"
            ),
            fields={
                **_control_loop_fields(round_report),
                "tool_calls": len(tool_names),
                "next_stage": "act" if tool_names else "finalize",
                "termination_reason": "tool_request" if tool_names else "no_tool_call",
            },
        )
        return {
            "messages": [message],
            "tool_rounds": state.tool_rounds + 1,
            # The offered set is carried, not recomputed. Selection can
            # involve a network-backed ranker, so recomputing it in `act`
            # could produce a different set and refuse a tool the model
            # was genuinely handed. An authority decision is made once.
            "metadata": {
                **state.metadata,
                "last_usage": result.usage.model_dump(),
                "offered_actions": offered,
            },
        }

    async def _call_model(
        self, request: CompletionRequest, state: AgentState, key: str
    ) -> CompletionResult:
        """One model call, protected and traced.

        The retry key includes the round so two genuinely different turns
        are not mistaken for a repeat of one — an idempotency key that is
        too coarse suppresses work that should happen.
        """

        async def call() -> CompletionResult:
            return await self._llm.complete(request)

        attempts_before = self._resilient.attempts
        try:
            result = await self._resilient.call(call, idempotency_key=key)
        except Exception as exc:
            error_fields = external_api_error_fields(exc)
            self._note_recovery(
                state,
                attempts_before,
                succeeded=False,
                error_fields=error_fields,
            )
            self._done_failed(
                "llm_call",
                summary="LLM call failed",
                event_kind="model",
                fields=error_fields,
            )
            if error_fields:
                self._logger.error("LLM API call failed", **error_fields)
            raise
        self._note_recovery(state, attempts_before, succeeded=True)
        self._account_for(result, state)
        return result

    def _note_recovery(
        self,
        state: AgentState,
        attempts_before: int,
        succeeded: bool,
        error_fields: dict[str, Any] | None = None,
    ) -> None:
        """Record whether the resilience patterns actually recovered anything.

        Only a call that needed more than one attempt counts: a call that
        succeeded first time neither attempted nor achieved a recovery,
        and counting it as a success would report a recovery rate of
        100% for a system that has never had to recover.
        """
        attempts = self._resilient.attempts - attempts_before
        if attempts <= 1:
            return
        metadata = state.metadata
        metadata["recoveries_attempted"] = int(metadata.get("recoveries_attempted", 0)) + 1
        if succeeded:
            metadata["recoveries_succeeded"] = int(metadata.get("recoveries_succeeded", 0)) + 1
        if self._reporter is not None:
            self._reporter_warning(
                "resilience",
                summary=("Recovered after retry" if succeeded else "Retry budget was exhausted"),
                event_kind="resilience",
                fields={
                    "attempts": attempts,
                    "recovered": succeeded,
                    **(error_fields or {}),
                },
            )

    def _account_for(self, result: CompletionResult, state: AgentState) -> None:
        """Trace what the call did.

        Tokens, not money: a rate card per provider is a second source of
        truth about pricing that goes stale silently, and the token count
        is what the harness actually observes.
        """
        if self._tracer is not None:
            # start/end rather than the context manager: those are the
            # Protocol's methods, so a LangSmith-backed tracer works here
            # without the graph knowing which one it was given.
            span = self._tracer.start_span(
                "llm_call",
                actor=state.actor,
                model_version=result.model,
            )
            self._tracer.end_span(span, result=result.finish_reason or "")
        self._record_slo("tool_failure_rate", 0.0)

    def _report_grounding(self, *, blocked: bool, summary: str, fields: dict[str, Any]) -> None:
        """Narrate the grounding stage — but only when there is one.

        With `hallucination_guard` withheld no detector is built, and
        `ResponseGuardrail.enforce` returns the draft unexamined. Printing
        "Grounded: 0/0 claims supported" for that would describe a check
        that never ran, which is worse than silence: it is the screen
        telling an operator a control is there when it is not. A stage
        that was not built has nothing to report.
        """
        if self._grounding is None:
            return
        report = self._reporter_blocked if blocked else self._done
        report("hallucination_check", summary=summary, event_kind="grounding", fields=fields)

    def _record_slo(self, metric: str, value: float) -> None:
        """Feed one observation to the SLO registry, when one is wired."""
        if self._slos is not None:
            self._slos.record(metric, value)

    async def act(self, state: AgentState) -> dict[str, Any]:
        """Execute the requested tool calls, each through the Gatekeeper."""
        last = state.messages[-1] if state.messages else Message()
        requests = last.tool_calls or []
        self._report(
            "tool_execution",
            summary=f"Executing {len(requests)} requested tool(s)",
            event_kind="stage",
            fields={"requested": len(requests)},
        )
        outputs: list[Message] = []
        results: list[str] = []
        decisions: list[str] = []
        invocation_fields_history: list[dict[str, Any]] = []
        offered = set(state.metadata.get("offered_actions", []))
        for requested in requests:
            self._done(
                "tool_request",
                event_kind="tool_request",
                summary=f"Requested: {requested.name}",
                fields=_safe_tool_request_fields(requested.name, requested.arguments),
            )
            text, decision, stage, invocation_fields = await self._invoke(
                requested.name, requested.arguments, state, offered
            )
            invocation_fields_history.append(invocation_fields)
            results.append(text)
            decisions.append(decision)
            self._note(state, f"result_of_{requested.name}", text, category="tool_result")
            outputs.append(
                Message(role="tool", content=text, tool_call_id=requested.id or requested.name)
            )

            # Once a proposal escalates, autonomous execution stops for the
            # remainder of this assistant turn. Do not execute later batched
            # side effects from the same model message.
            gatekeeper = invocation_fields.get("gatekeeper")
            if isinstance(gatekeeper, dict) and gatekeeper.get("decision") == "escalate":
                remaining = len(requests) - len(invocation_fields_history)
                if remaining:
                    self._reporter_blocked(
                        "gatekeeper_evaluation",
                        summary=f"Skipped {remaining} tool proposal(s) after escalation",
                        event_kind="gatekeeper",
                        fields={
                            "decision": "escalate",
                            "reason_code": "AUTONOMY_STOPPED_AFTER_ESCALATION",
                            "skipped_requests": remaining,
                        },
                    )
                break

        self._done(
            "tool_execution",
            event_kind="tool_result",
            summary=f"Processed {len(requests)} requested tool(s)",
            fields={
                "requested": len(requests),
                "completed": sum(1 for item in invocation_fields_history if item.get("tool_status") == "completed"),
                "blocked": sum(1 for item in invocation_fields_history if item.get("tool_status") == "blocked"),
                "failed": sum(1 for item in invocation_fields_history if item.get("tool_status") == "failed"),
            },
        )
        turn_escalated = any(
            item.get("gatekeeper", {}).get("decision") == "escalate"
            for item in invocation_fields_history
            if isinstance(item.get("gatekeeper"), dict)
        )
        authoritative_results = [
            text
            for text, invocation in zip(results, invocation_fields_history, strict=True)
            if invocation.get("tool_status") == "completed"
        ]
        return {
            "messages": outputs,
            "tool_results": results,
            "authoritative_tool_results": [
                *state.authoritative_tool_results,
                *authoritative_results,
            ],
            "decisions": decisions,
            "escalated": state.escalated or turn_escalated,
            "metadata": {
                **state.metadata,
                "last_tool_invocations": invocation_fields_history,
            },
        }

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """One tool-free call so the customer always gets an answer.

        Without this, hitting the tool-iteration cap returns whatever the
        last turn happened to contain — often nothing. A cap is a
        control, not an excuse to hand the customer an empty reply.
        """
        self._report(
            "finalize",
            summary="Preparing the customer-facing response",
        )
        # A previous model call can already have exhausted its retry budget.
        # Do not immediately spend another retry budget on finalize; surface
        # the outage and hand the turn to the human-facing fallback.
        if state.metadata.get("external_dependency_failed"):
            return {
                "response": _FALLBACK_REPLY,
                "decisions": [*state.decisions, "resilience: llm unavailable after retries"],
                "escalated": True,
                "metadata": state.metadata,
            }
        last = state.messages[-1] if state.messages else Message()
        authoritative_evidence = list(state.authoritative_tool_results)
        critique = self._deliberation.reflect(last.content, authoritative_evidence)
        reflection_comment = critique.summary
        self._logger.info(
            "Customer response reflection completed",
            acceptable=critique.acceptable,
            problems=critique.problems,
            evidence_count=len(authoritative_evidence),
        )
        # Reflection is part of the control loop, not a standalone workflow
        # stage. Only expose an actual reflector comment from the current
        # Critique; do not manufacture a success message for the TUI.
        if reflection_comment:
            reflection_event_fields = {"reflection_comment": reflection_comment}
            if critique.acceptable:
                self._done(
                    "control_loop",
                    event_kind="control_loop",
                    fields=reflection_event_fields,
                )
            else:
                self._reporter_warning(
                    "control_loop",
                    event_kind="control_loop",
                    fields=reflection_event_fields,
                )
        if (
            last.role == "assistant"
            and not last.tool_calls
            and is_customer_ready(last.content)
            and not self._is_deliberation(last.content)
            # Reflection is what decides whether the draft in hand is good
            # enough to send, or whether the second call is worth its cost.
            # Without it the only question asked was "is this shaped like
            # a reply", which a confident wrong answer also passes.
            and critique.acceptable
        ):
            # Grounding runs on this path too. It used to be checked only
            # when a second call was needed, which is the *rarer* path —
            # so the reply that usually reached the customer was the one
            # never checked.
            evidence = [
                Chunk(content=text, kind="tool_result")
                for text in authoritative_evidence
            ]
            guardrail_result = await self._response_guardrail.enforce(last.content, evidence)
            guardrail_fields = {
                **guardrail_result.model_dump(exclude={"response", "unsupported_claims"}),
                "schema_valid": True,
                "unsupported_claims": guardrail_result.unsupported_claims,
            }
            self._report_grounding(
                blocked=guardrail_result.escalated,
                summary=(
                    "LLM hallucinated"
                    if guardrail_result.escalated
                    else f"Grounded: {guardrail_result.supported}/"
                    f"{guardrail_result.claims} claims supported"
                ),
                fields=guardrail_fields,
            )
            return {
                "response": guardrail_result.response,
                "decisions": [
                    *state.decisions,
                    f"grounding: {guardrail_result.reason_code}",
                ],
                "escalated": state.escalated or guardrail_result.escalated,
            }
        # If reflection found an issue and there is no authoritative tool
        # evidence, run the deterministic hallucination gate before asking
        # another LLM to rewrite the answer. Otherwise the repair model can
        # manufacture a plausible business outcome from the same unsupported
        # draft, which is exactly the 0/0 grounding failure this gate prevents.
        if not critique.acceptable and not authoritative_evidence:
            evidence = []
            guardrail_result = await self._response_guardrail.enforce(
                last.content,
                evidence,
            )
            if guardrail_result.escalated:
                guardrail_fields = guardrail_result.model_dump(
                    exclude={"response"},
                )
                self._report_grounding(
                    blocked=True,
                    summary=(
                        "Response blocked before LLM repair · no authoritative tool evidence"
                    ),
                    fields=guardrail_fields,
                )
                return {
                    "response": guardrail_result.response,
                    "decisions": [
                        *state.decisions,
                        f"reflection: rejected ({critique.summary})",
                        f"grounding: {guardrail_result.reason_code}",
                    ],
                    "escalated": state.escalated or guardrail_result.escalated,
                }

        # The reflector already logged what it found; saying it twice
        # here would double every rejection in the trace.
        self._report(
            "llm_call",
            event_kind="model",
            summary="Generating the final customer response",
            fields={"purpose": "reply_revision"},
        )
        try:
            result = await self._call_model(
                CompletionRequest(
                    messages=[
                        *self._prompt_messages(state),
                        Message(role="user", content=build_reply_instruction()),
                    ],
                    json_schema=CustomerResponse.model_json_schema(),
                ),
                state,
                f"{state.session_id}:finalize",
            )
        except Exception as exc:
            error_fields = external_api_error_fields(exc)
            self._reporter_warning(
                "llm_call",
                summary="LLM unavailable after retries",
                event_kind="resilience",
                fields={
                    "dependency": self._resilient.name,
                    **(error_fields or {}),
                    **({"reason": str(exc)} if not error_fields else {}),
                },
            )
            return {
                "response": _FALLBACK_REPLY,
                "decisions": [*state.decisions, "resilience: llm unavailable after retries"],
                "escalated": True,
                "metadata": {**state.metadata, "external_dependency_failed": True},
            }
        self._done(
            "llm_call",
            tokens=result.usage.total_tokens,
            summary="Final customer response generated",
            event_kind="model",
            fields={
                "purpose": "reply_revision",
                "model": result.model,
                "finish_reason": result.finish_reason or "unknown",
                "input_tokens": result.usage.prompt_tokens,
                "output_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        )
        reply, schema_fields = self._response_guardrail.validate_schema(result.content)
        if reply is None or not is_customer_ready(reply):
            reply = _FALLBACK_REPLY
            schema_fields["schema_valid"] = False

        async def revise_unsupported(candidate: str, unsupported: list[str]) -> str:
            self._report(
                "hallucination_repair",
                event_kind="grounding",
                summary="Revising unsupported claims against tool evidence",
                fields={"unsupported": unsupported, "attempt": 1},
            )
            repair_prompt = (
                "Rewrite the customer response so that every factual claim is "
                "strictly supported by the tool results below. Remove any "
                "unsupported detail. Do not invent facts. Return only JSON "
                "matching the response schema.\n\n"
                f"Draft:\n{candidate}\n\n"
                "Authoritative tool results:\n"
                + "\n".join(f"- {item}" for item in authoritative_evidence)
            )
            repaired_result = await self._call_model(
                CompletionRequest(
                    messages=[Message(role="user", content=repair_prompt)],
                    json_schema=CustomerResponse.model_json_schema(),
                ),
                state,
                f"{state.session_id}:hallucination-repair",
            )
            repaired, _ = self._response_guardrail.validate_schema(repaired_result.content)
            return repaired or _FALLBACK_REPLY

        evidence = [
            Chunk(content=text, kind="tool_result")
            for text in authoritative_evidence
        ]
        guardrail_result = await self._response_guardrail.enforce(
            reply,
            evidence,
            revise=revise_unsupported,
        )
        reply = guardrail_result.response
        guardrail_fields = guardrail_result.model_dump(
            exclude={"response"},
        )
        self._report_grounding(
            blocked=guardrail_result.escalated,
            summary=(
                "Response blocked by hallucination guard"
                if guardrail_result.escalated
                else f"Grounded: {guardrail_result.supported}/"
                f"{guardrail_result.claims} claims supported"
            ),
            fields=guardrail_fields,
        )
        state.metadata["hallucination_rate"] = guardrail_result.hallucination_rate
        state.metadata["guardrail_reason_code"] = guardrail_result.reason_code
        state.metadata["guardrail_escalated"] = guardrail_result.escalated
        grounding = [
            (
                f"grounding: {guardrail_result.reason_code} "
                f"claims={guardrail_result.claims} "
                f"supported={guardrail_result.supported} "
                f"unsupported={guardrail_result.unsupported}"
            )
        ]
        decisions = [*state.decisions, *grounding]
        if not critique.acceptable:
            decisions.append(f"reflection: revised ({critique.summary})")
        if guardrail_result.escalated:
            decisions.append(
                f"guardrail: escalated ({', '.join(guardrail_result.failure_kinds) or 'unsupported'})"
            )
        # Scored by something independent of what wrote it. One bounded
        # pass: the loop revises once and reports, rather than iterating
        # towards a sentence nobody is waiting for.
        score = await self._deliberation.score(reply, authoritative_evidence)
        if authoritative_evidence:
            decisions.append(f"evaluation: {score:.2f} supported")
        self._done(
            "finalize",
            summary="Customer response ready",
            fields={
                "response_ready": True,
                "revised": not critique.acceptable,
                "evaluation_score": score if authoritative_evidence else None,
                "grounding_issues": guardrail_result.unsupported,
                "schema_valid": schema_fields.get("schema_valid", True),
                "validation_mode": schema_fields.get("validation_mode"),
                "validation_attempts": schema_fields.get("validation_attempts"),
                "repair_attempts": guardrail_result.repair_attempts,
                "grounded": guardrail_result.grounded,
                "hallucination_scope": guardrail_result.hallucination_scope,
                "enforcement": "ESCALATE" if guardrail_result.escalated else "PASS",
                "failure_kinds": guardrail_result.failure_kinds,
                "nollm_resolved_ratio": round(guardrail_result.nollm_resolved_ratio, 3),
                "tier_counts": guardrail_result.tier_counts,
                "guardrail_reason_code": guardrail_result.reason_code,
            },
        )
        return {
            "response": reply,
            "messages": [Message(role="assistant", content=reply)],
            "decisions": decisions,
        }

    def _system_prompt(self) -> str:
        """The active prompt, versioned in procedural memory.

        Read through the store rather than from the module so that the
        prompt this agent is running has a version and a history —
        rolling back becomes selecting an earlier version rather than
        finding the commit that changed a string.
        """
        procedures = self._memory.procedural if self._memory is not None else None
        return build_system_prompt(procedures)

    def _is_deliberation(self, content: str) -> bool:
        """Whether this text is the model thinking rather than replying.

        A reply addresses the customer; deliberation addresses the model
        itself, or refers to the customer in the third person. Shipping
        the latter reads as "We need to respond as per the conversation",
        which is the model's notes, not an answer.
        """
        lowered = content.casefold()
        return any(marker in lowered for marker in self._deliberation_markers)

    # ---- wiring ------------------------------------------------------

    def _build(self) -> Any:
        """Assemble the state graph from the stages this agent was given.

        Only enabled stages become nodes, and the edges are wired to the
        next enabled one — so a withheld capability is absent from the
        compiled graph rather than present and declining to act. That is
        the difference an episode is trying to show: the trace has a gap
        where the control used to be, not a step reporting that it did
        nothing.
        """
        graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
        graph.add_node("authenticate", self.authenticate)
        graph.add_node("choose_order", self.choose_order)
        graph.add_node("read_intent", self.read_intent)
        graph.add_node("reason", self.reason)
        graph.add_node("act", self.act)
        graph.add_node("finalize", self.finalize)

        # The screening-and-routing run, in order, each present only if
        # this deployment declared it.
        front_door = [
            stage
            for stage in (
                # Deliberately first: deterministic domain contradictions are
                # settled from the database before anything spends a model call.
                ("basic_sanity", self.basic_sanity),
                ("prompt_injection", self.prompt_injection),
                ("content_moderation", self.content_moderation),
                ("classify_intent", self.classify_intent),
                ("assemble_context", self.assemble_context),
            )
            if self._features.is_on(_STAGE_FEATURES[stage[0]])
            and self._has_collaborator(stage[0])
        ]
        for name, node in front_door:
            graph.add_node(name, node)

        # Authentication is the hard front-door gate.  An unauthenticated
        # request never reaches security screening, intent classification,
        # retrieval, order selection, or the LLM.
        graph.set_entry_point("authenticate")
        graph.add_conditional_edges(
            "authenticate",
            self._after_authenticate,
            {"choose_order": "choose_order", "done": END},
        )
        # Resolve deterministic menu choices before anything model-driven,
        # then hand to the first stage this agent actually has.
        names = [name for name, _ in front_door]
        first = names[0] if names else "reason"
        graph.add_conditional_edges(
            "choose_order", self._after_choose, {"next": first, "done": END}
        )

        # Each stage continues to the next one that exists. Every one of
        # them can halt the turn, so each edge carries a `done` branch.
        for position, name in enumerate(names):
            following = names[position + 1] if position + 1 < len(names) else "reason"
            if name == "classify_intent":
                graph.add_conditional_edges(
                    name,
                    self._after_classify_intent,
                    {"read_intent": "read_intent", "next": following, "done": END},
                )
            else:
                graph.add_conditional_edges(
                    name, self._after_screening, {"next": following, "done": END}
                )

        graph.add_edge("read_intent", END)
        graph.add_conditional_edges(
            "reason", self._after_reason, {"act": "act", "finalize": "finalize"}
        )
        graph.add_conditional_edges(
            "act", self._after_act, {"reason": "reason", "finalize": "finalize"}
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    def _has_collaborator(self, stage: str) -> bool:
        """Whether the object a stage needs was actually injected.

        Belt and braces for a caller that constructs the agent directly:
        a declared feature whose collaborator is missing would otherwise
        compile a node that raises on its first turn.
        """
        required = {"basic_sanity": self._basic_sanity, "assemble_context": self._context}
        return required.get(stage, self) is not None

    def _after_screening(self, state: AgentState) -> str:
        """A screening stage either halted the turn or passed it along."""
        return "done" if state.metadata.get("halted") else "next"

    def _after_classify_intent(self, state: AgentState) -> str:
        """Route reads deterministically; keep writes and ambiguity on the LLM path."""
        if state.metadata.get("halted"):
            return "done"
        if state.metadata.get("intent_execution_route") == "deterministic_read":
            return "read_intent"
        return "next"

    def _after_authenticate(self, state: AgentState) -> str:
        """Stop until identity is verified; only verified sessions enter intake.

        A turn that newly supplies a valid customer code ends after the
        authentication read-back.  The next customer turn then enters the
        deterministic order/menu and sanity path before security/intent.
        """
        return "done" if state.metadata.get("halted") else "choose_order"

    def _after_choose(self, state: AgentState) -> str:
        """A menu choice is answered from the database and finishes here."""
        return "done" if state.metadata.get("halted") else "next"

    def _after_reason(self, state: AgentState) -> str:
        """Act on requested tools, or finish."""
        last = state.messages[-1] if state.messages else Message()
        return "act" if last.tool_calls else "finalize"

    def _after_act(self, state: AgentState) -> str:
        """Continue only while autonomous execution remains permitted."""
        # A Gatekeeper ESCALATE is a terminal autonomous decision for this turn.
        # Do not send the model back to reason(), or it could propose another
        # side effect after the system has explicitly handed control to a human.
        if state.escalated:
            return "finalize"
        next_round = state.tool_rounds + 1
        if next_round <= self._max_rounds:
            return "reason"
        report = self._control_loop.evaluate_round(next_round, self._max_rounds)
        self._reporter_blocked(
            "control_loop",
            summary="Maximum tool iterations reached",
            event_kind="control_loop",
            fields=_control_loop_fields(report),
        )
        return "finalize"

    async def _execute_ungoverned(
        self,
        call: Any,
        authority: AuthorityContext,
        control_fields: dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Run a tool with nothing standing between the model and execution.

        Reachable only where the `gatekeeper` feature has been withheld,
        which is what an episode does to show the difference. Two things
        still hold: the audit entry is written, so the trail shows that an
        unauthorized execution happened rather than showing nothing at
        all; and no budget is committed, because no budget was consulted.
        """
        # No report. A stage that was not built has nothing to say, and a
        # GATEKEEPER line reading "no gatekeeper" is the screen describing
        # a control that is not there — the absence is the demonstration,
        # so the trace should show a gap where the check used to be. The
        # audit entry below still records the ungoverned execution.
        approved = ApprovedToolCall.ungoverned(call, authority)
        self._audit.record(
            AuditEntry(
                trace_id=str(authority.request_context.get("metadata", {}).get("trace_id", "")),
                actor=authority.actor,
                action=call.action,
                authority_context=authority.model_dump(mode="json"),
                decision="allow",
                rationale=approved.decision.reason,
            )
        )
        # Straight to the agent's domain adapter, with no execution
        # boundary around it: no idempotency key, so asking twice really
        # does refund twice, and no timeout, so a slow tool holds the turn
        # open. Those are the Executor's guarantees, and this is what the
        # absence of them looks like.
        outcome = await self._execute_directly(call)
        return self._after_execution(call, outcome, control_fields, gatekeeper_fields={})

    async def _execute_directly(self, call: Any) -> ExecutionOutcome:
        """Run one tool with nothing wrapped around it.

        Shaped as an `ExecutionOutcome` so the reporting path downstream
        cannot tell the difference — what changed is that no idempotency
        key was taken and no deadline applied, not how the result reads.
        `replayed` is False by construction here: replay is exactly the
        thing an idempotency guard provides, and there is none.
        """
        started = utc_now()
        if self._handler is None:
            result = ToolResult(success=False, error="No tool handler is wired for direct execution")
        else:
            result = await self._handler.execute(call)
        return ExecutionOutcome(
            result=result,
            status="completed" if result.success else "failed",
            replayed=False,
            started_at=started,
            ended_at=utc_now(),
        )

    async def _invoke(
        self, name: str, arguments: str, state: AgentState, offered: set[str]
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Validate, control, authorize, approve and execute one tool call."""
        call = self._registry.build_call(name, arguments)
        if call is None:
            blocked_fields = {
                "tool": name,
                "result": "invalid_request",
                "reason_code": "TOOL_CALL_INVALID",
                "reason": "tool is unknown or arguments failed validation",
            }
            self._reporter_blocked(
                "tool_result",
                summary=f"Tool request rejected: {name}",
                event_kind="tool_result",
                fields=blocked_fields,
            )
            return (
                f"Tool '{name}' is unavailable or its arguments were invalid.",
                f"{name}: invalid",
                "tool_refused",
                {"tool_result": blocked_fields, "tool_status": "blocked"},
            )

        call = call.model_copy(update={"params": self._security.minimize(call.params)})
        authority = AuthorityContext(
            actor=state.actor or self._actor,
            granted_by="user_task",
            session_id=state.session_id,
            offered_actions=sorted(offered),
            request_context={
                "query": state.query,
                "customer_id": state.customer_id,
                "metadata": dict(state.metadata),
                "tool_action": call.action,
                "tool_params": dict(call.params),
            },
        )

        control_report = self._control_loop.before_action(
            call, round_number=state.tool_rounds, max_rounds=self._max_rounds
        )
        control_fields = _control_loop_fields(control_report)
        if control_report.decision == "abort":
            self._reporter_blocked(
                "control_loop",
                summary=(
                    "Duplicate action suppressed"
                    if control_report.reason_code == "DUPLICATE_ACTION"
                    else "Control loop aborted the action"
                ),
                event_kind="control_loop",
                fields=control_fields,
            )
            result_fields = {
                "tool": call.action,
                "result": (
                    "suppressed_duplicate"
                    if control_report.reason_code == "DUPLICATE_ACTION"
                    else "control_loop_aborted"
                ),
                "execution_status": "not_executed",
            }
            self._reporter_blocked(
                "tool_result",
                summary=f"Tool not executed: {call.action}",
                event_kind="tool_result",
                fields=result_fields,
            )
            return (
                self._control_loop.reject_message(call)
                if control_report.reason_code == "DUPLICATE_ACTION"
                else "The control loop stopped this action before execution.",
                (
                    f"{call.action}: suppressed (duplicate)"
                    if control_report.reason_code == "DUPLICATE_ACTION"
                    else f"{call.action}: control loop aborted"
                ),
                "tool_suppressed",
                {
                    "control_loop": control_fields,
                    "control_loop_status": "blocked",
                    "tool_result": result_fields,
                    "tool_status": "blocked",
                },
            )

        if self._gatekeeper is None:
            return await self._execute_ungoverned(call, authority, control_fields)

        self._report(
            "gatekeeper_evaluation",
            event_kind="gatekeeper",
            summary="Checking deterministic execution admissibility",
            fields={"requested_action": call.action},
        )
        decision = await self._gatekeeper.evaluate(call, authority)
        gatekeeper_fields = _gatekeeper_fields(decision, call)
        if not decision.allowed:
            self._reporter_blocked(
                "gatekeeper_evaluation",
                summary=f"Gatekeeper: {decision.verdict.upper()}",
                event_kind="gatekeeper",
                fields=gatekeeper_fields,
            )
            if decision.question == "offered":
                self._logger.warning("Model called a tool it was not offered", tool=call.action)
                text = f"'{call.action}' is not available for this request."
                result_fields = {
                    "tool": call.action,
                    "result": "not_offered",
                    "execution_status": "not_executed",
                }
            elif decision.verdict == "escalate":
                text = f"That action requires human review: {decision.reason}"
                result_fields = {
                    "tool": call.action,
                    "result": "human_review_required",
                    "execution_status": "not_executed",
                }
            else:
                text = f"That action was not permitted: {decision.reason}"
                result_fields = {
                    "tool": call.action,
                    "result": "not_executed",
                    "execution_status": "not_executed",
                }
            self._reporter_blocked(
                "tool_result",
                summary=f"Tool not executed: {call.action}",
                event_kind="tool_result",
                fields=result_fields,
            )
            return (
                text,
                f"{call.action}: {decision.verdict} ({decision.question}) — {decision.reason}",
                "tool_refused",
                {
                    "control_loop": control_fields,
                    "control_loop_status": "completed",
                    "gatekeeper": gatekeeper_fields,
                    "gatekeeper_status": "blocked",
                    "tool_result": result_fields,
                    "tool_status": "blocked",
                },
            )

        self._done(
            "gatekeeper_evaluation",
            event_kind="gatekeeper",
            summary="Gatekeeper: ALLOW",
            fields=gatekeeper_fields,
        )
        approved = ApprovedToolCall.from_decision(call, decision, authority)
        outcome = await self._executor.execute(approved)
        if outcome.result.success and not outcome.replayed:
            # Budget is charged only for a real completion, not a replay.
            self._gatekeeper.commit(call, authority)
        return self._after_execution(call, outcome, control_fields, gatekeeper_fields)

    def _after_execution(
        self,
        call: Any,
        outcome: Any,
        control_fields: dict[str, Any],
        gatekeeper_fields: dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Report one completed tool call, governed or not."""
        result = outcome.result
        result_fields = _safe_tool_result_fields(call, result, outcome=outcome)
        if result.success:
            self._control_loop.record_success(call, result.output)
            self._done(
                "tool_result",
                event_kind="tool_result",
                summary=f"Tool completed: {call.action}",
                fields=result_fields,
            )
            tool_status = "completed"
        else:
            self._done_failed(
                "tool_result",
                summary=f"Tool failed: {call.action}",
                event_kind="tool_result",
                fields=result_fields,
            )
            tool_status = "failed"
        return (
            result.output if result.success else f"Tool failed: {result.error}",
            f"{call.action}: {'ok' if result.success else 'failed'}",
            "tool_execution" if result.success else "tool_refused",
            {
                "control_loop": control_fields,
                "control_loop_status": "completed",
                "gatekeeper": gatekeeper_fields,
                "gatekeeper_status": "completed" if gatekeeper_fields else "absent",
                "executor": result_fields,
                "tool_result": result_fields,
                "tool_status": tool_status,
            },
        )

    async def _offered_tools(self, state: AgentState) -> list[Any]:
        """Which tools this turn may use.

        Two independent narrowings, and both matter: the injection guard
        removes what untrusted content must not reach, and retrieval
        removes what this task does not need. Neither substitutes for the
        Gatekeeper, which still rules on whatever is actually called.
        """
        allowed = set(state.metadata.get("allowed_actions", []))
        classified = "intent" in state.metadata
        if not classified or state.metadata.get("intent_execution_route") == "llm_fallback":
            # Either the classifier could not safely narrow the capability, or
            # this agent has no classifier at all. Both mean *nothing narrowed
            # by intent* — which is not the same as nothing permitted. Treating
            # the absent classifier as an empty permission set offered the model
            # no tools whatsoever, so `/harness off` produced an agent that
            # could not act rather than one free to act: the exact inverse of
            # what switching the harness off is meant to show.
            #
            # This grants no authority. The Gatekeeper and RBAC still rule on
            # every invocation, and the injection guard's narrowing still holds.
            return [
                tool
                for tool in self._registry.all_tools()
                if not allowed or tool.qualified_name in allowed
            ]

        by_intent = set(state.metadata.get("intent_tools", []))
        if not by_intent:
            return []
        tools = [
            tool
            for tool in self._registry.all_tools()
            if tool.name in by_intent and (not allowed or tool.qualified_name in allowed)
        ]
        if self._selector is None:
            return tools
        # Retrieval narrows further *within* what the intent permits. It
        # can never widen it, and an empty result means no tools rather
        # than every tool.
        # Deterministic plan steps are mandatory execution prerequisites.
        # Retrieval may narrow optional tools, but it must never remove a
        # tool explicitly required by the trusted intent plan.
        plan_tools = {
            step.get("tool")
            for step in state.metadata.get("plan", [])
            if isinstance(step, dict) and step.get("tool")
        }

        mandatory = [
            tool
            for tool in tools
            if tool.name in plan_tools
        ]

        selected = await self._selector.select(state.query)
        narrowed = [tool for tool in selected if tool in tools]

        mandatory_names = {tool.qualified_name for tool in mandatory}
        return (
            mandatory
            + [
                tool
                for tool in narrowed
                if tool.qualified_name not in mandatory_names
            ]
            or mandatory
            or tools
        )

    def _prompt_messages(self, state: AgentState) -> list[Message]:
        """System prompt, assembled context, the query, and the turns so far."""
        system = self._system_prompt()
        if state.context:
            system = f"{system}\n\n# Retrieved context\n{state.context}"
        if state.metadata.get("intent_execution_route") == "llm_fallback":
            prior_observations = self._authoritative_history.get(state.session_id, [])
            prior_context = (
                "\n\n# Prior authoritative tool observations\n"
                + "\n".join(f"- {item}" for item in prior_observations[-8:])
                if prior_observations
                else ""
            )
            system = (
                f"{system}{prior_context}\n\n"
                "# Intent fallback\n"
                "The lightweight intent classifier could not resolve this turn. "
                "Use the full conversation context, prior authoritative tool "
                "observations, the current user message, and any current-turn "
                "tool results to infer what the customer means. Prior tool "
                "observations are context only; they do not prove that a new "
                "action happened in this turn. You may ask a clarifying "
                "question when context is still insufficient. Never claim "
                "that a refund, approval, queueing, cancellation, or other "
                "business action occurred unless you actually call the "
                "corresponding tool and receive a successful tool result.\n"
            )
        return [
            Message(role="system", content=system),
            *self.history_for(state.session_id),
            Message(role="user", content=state.query),
            *state.messages,
        ]

    def _report(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: str = "stage",
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

    def _done_failed(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: str = "stage",
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

    def _done(
        self,
        stage: str,
        tokens: int | None = None,
        detail: str | None = None,
        *,
        event_kind: str = "stage",
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

    def _reporter_blocked(
        self,
        stage: str,
        *,
        detail: str | None = None,
        summary: str | None = None,
        event_kind: str = "decision",
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
        event_kind: str = "system",
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
        event_kind: str = "stage",
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


def _safe_tool_request_fields(name: str, arguments: str) -> dict[str, Any]:
    """Expose only small, domain-relevant tool arguments to the TUI."""
    display_name = _display_tool_name(name)
    fields: dict[str, Any] = {
        "tool": display_name,
        "qualified_tool": name.replace("__", "."),
    }
    try:
        raw = json.loads(arguments or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return fields
    if not isinstance(raw, dict):
        return fields
    for key in ("order_id", "amount"):
        value = raw.get(key)
        if isinstance(value, (str, int, float)):
            fields[key] = f"${value:,.2f}" if key == "amount" and isinstance(value, (int, float)) else value
    return fields


def _basic_sanity_fields(result: Any) -> dict[str, Any]:
    """Render BasicSanity findings consistently with the TUI check contract."""
    fields: dict[str, Any] = {
        "passed": result.passed,
        "reason_code": result.reason_code,
        "checks": [],
    }
    facts = result.facts if isinstance(result.facts, dict) else {}
    for key in (
        "order_id",
        "customer_id",
        "order_status",
        "refund_status",
        "amount",
        "requested_amount",
        "already_refunded",
        "refund_eligible",
        "operation",
        "order_id_present",
        "lookup_attempted",
    ):
        if key in facts:
            fields[key] = facts[key]
    for finding in result.findings:
        evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
        status = (
            "SKIPPED"
            if not finding.applicable
            else "PASS"
            if finding.passed
            else "FAIL"
        )
        fields["checks"].append(
            {
                "name": finding.check.replace("_", " ").title(),
                "status": status,
                "reason_code": (
                    f"BASIC_SANITY_{finding.check.upper()}_DEFERRED"
                    if not finding.applicable
                    else f"BASIC_SANITY_{finding.check.upper()}_PASS"
                    if finding.passed
                    else result.reason_code
                ),
                "reason": finding.reason,
                "observed": evidence,
            }
        )
    if not fields["checks"]:
        fields["checks"].append(
            {
                "name": "Basic sanity",
                "status": "PASS" if result.passed else "FAIL",
                "reason_code": result.reason_code,
                "reason": result.message or result.reason_code,
                "observed": facts,
            }
        )
    return fields


def _control_loop_fields(report: ControlLoopReport) -> dict[str, Any]:
    """Render only the active control-loop decision and its evidence."""
    fields: dict[str, Any] = {
        "decision": report.decision,
        "reason_code": report.reason_code,
        "checks": [check.model_dump(mode="json") for check in report.checks],
    }
    # The fingerprint is only useful when diagnosing duplicate suppression.
    if report.fingerprint:
        fields["fingerprint"] = report.fingerprint
    return fields


def _gatekeeper_fields(decision: Any, call: Any) -> dict[str, Any]:
    """Expose the exact proposal and deterministic checks behind the ruling."""
    fields: dict[str, Any] = {
        "decision": decision.verdict,
        "action": _display_tool_name(decision.action),
        "effect": "write" if call.side_effecting else "read",
        "reason_code": decision.reason_code,
        "question": decision.question,
        "checks": [check.model_dump(mode="json") for check in decision.checks],
        "authority_rank": decision.authority_rank,
        "required_authority_rank": decision.required_authority_rank,
        "max_granting_rank": decision.max_granting_rank,
        "cumulative_total": decision.cumulative_total,
        "threshold": decision.threshold,
    }
    order_id = call.params.get("order_id")
    if isinstance(order_id, (str, int)) and not isinstance(order_id, bool):
        fields["order_id"] = order_id
    amount = call.params.get("amount")
    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
        fields["amount"] = float(amount)

    for check in decision.checks:
        if check.check_id == "gatekeeper.actor_authorization":
            permission = check.observed.get("required_permission")
            role = check.observed.get("role")
            if permission:
                fields["permission"] = permission
            if role:
                fields["role"] = role
            break

    return fields

def _extract_order_id(message: str) -> int | None:
    """Extract an order id only when it is anchored to the word "order"."""
    match = re.search(
        r"\border\s*(?:#|no\.?|number)?\s*(\d{4,})\b|\b(\d{4,})\s+order\b",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _display_tool_name(name: str) -> str:
    """Return the concise tool name used in the operator-facing dashboard."""
    normalized = name.replace("__", ".")
    return normalized.rsplit(".", maxsplit=1)[-1]


def _safe_tool_result_fields(call: Any, result: Any, *, outcome: Any | None = None) -> dict[str, Any]:
    """Expose domain outcome plus generic Executor lifecycle metadata."""
    fields: dict[str, Any] = {
        "tool": _display_tool_name(call.action),
        "qualified_tool": call.action,
        "result": "success" if result.success else "failed",
    }
    if outcome is not None:
        fields.update(
            {
                "execution_status": outcome.status,
                "replayed": outcome.replayed,
                "timed_out": outcome.timed_out,
                "duration_ms": round(outcome.duration_ms, 2),
                "idempotency_key": outcome.idempotency_key,
            }
        )
    for key in (
        "order_id",
        "amount",
        "order_status",
        "refund_status",
        "pending_review",
        "queued",
        "already_queued",
        "request_id",
        "refund_id",
        "steps",
        "failed_step",
        "compensated",
    ):
        value = result.data.get(key)
        if value is None:
            continue
        if key == "amount" and isinstance(value, (int, float)):
            fields[key] = f"${float(value):,.2f}"
        elif isinstance(value, (str, int, float, bool)):
            fields[key] = value
        elif isinstance(value, list):
            fields[key] = len(value)
    if not result.success and result.error:
        fields["error"] = result.error
    return fields


# A customer asking to see the list again. Deterministic, like the list
# itself — re-reading a table is not a task for a model.
_LIST_REQUEST = re.compile(
    r"\b(my orders|all my orders|list (my )?orders|show (me )?(my )?orders|"
    r"what orders|which orders|see (my )?orders|orders again|the list)\b",
    re.IGNORECASE,
)


def _asks_for_the_list(message: str) -> bool:
    """Whether the customer is asking to see their order list again."""
    return bool(_LIST_REQUEST.search(message))


def is_customer_ready(content: str) -> bool:
    """Whether this text can be shown to a customer as an answer.

    A weak model sometimes emits a tool call as plain *text* instead of a
    structured call — `{"name": "get_order_status", "parameters": {...}}`.
    That is machine output that failed to become a tool call, and handing
    it to a customer is worse than saying nothing. It is caught here by
    shape rather than by pattern-matching any one model's phrasing.
    """
    text = content.strip()
    if not text:
        return False
    if not (text.startswith(("{", "[")) and text.endswith(("}", "]"))):
        return True
    try:
        json.loads(text)
    except ValueError:
        # Not valid JSON, so it is prose that merely contains braces.
        return True
    # It parsed. Whatever its keys, a bare JSON document is machine
    # output that failed to become a tool call — checking for particular
    # key names only catches the spellings you thought of, and the live
    # failures used three different ones.
    return False


def parse_arguments(raw: str) -> dict[str, Any]:
    """Parse a model's tool arguments, tolerating an empty string."""
    if not raw.strip():
        return {}
    parsed: dict[str, Any] = json.loads(raw)
    return parsed
