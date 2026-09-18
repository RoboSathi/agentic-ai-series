"""The LangGraph refund agent (LLD §21, HLD §4 request lifecycle).

Nodes, in order:

    intake → assemble_context → reason ⇄ act → finalize

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
from typing import Any

from langgraph.graph import END, StateGraph

from agentic_commons.context import (
    Chunk,
    ContextFailureDetector,
    FiveStagePipeline,
    StrategySelector,
    Task,
)
from agentic_commons.evals import NollmScoringCascadeImpl
from agentic_commons.gatekeeper import AuthorityContext, GatekeeperImpl
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
from agentic_commons.resilience import ResilientCaller
from agentic_commons.security import SecuritySuite
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from customer_support_agent.config import get_agent_settings
from customer_support_agent.control_loop import (
    CascadeEvaluator,
    DeliberationLoop,
    DuplicateActionDetector,
)
from customer_support_agent.core.accounting import TurnAccountant, TurnOutcome
from customer_support_agent.core.authentication import CustomerAuthenticator
from customer_support_agent.core.identity import (
    Customer,
    current_customer,
    set_current_customer,
)
from customer_support_agent.core.intent import Intent, IntentClassifier
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
from customer_support_agent.core.scope import ScopeGuard
from customer_support_agent.core.tools import SupportTools
from customer_support_agent.prompts.system_prompt import (
    build_reply_instruction,
    build_system_prompt,
)

# Shown when the model produces nothing a customer could read. Rare, and
# an honest hand-off beats a confident non-answer.
_FALLBACK_REPLY = (
    "Sorry — I wasn't able to complete that. Let me pass you to a colleague who can help."
)


class CustomerSupportAgent:
    """The refund agent: a graph plus the subsystems its nodes call."""

    def __init__(
        self,
        llm: LLMClient,
        gatekeeper: GatekeeperImpl,
        registry: NamespacedToolRegistry,
        tools: SupportTools,
        security: SecuritySuite,
        memory: MemoryFacade | None = None,
        context: FiveStagePipeline | None = None,
        selector: ToolRetrievalSelector | None = None,
        cascade: NollmScoringCascadeImpl | None = None,
        reporter: StageReporter | None = None,
        resilient: ResilientCaller | None = None,
        tracer: Tracer | None = None,
        slos: SLORegistry | None = None,
        deliberation: DeliberationLoop | None = None,
        grounding: DetectionLadder | None = None,
        authenticator: CustomerAuthenticator | None = None,
        orders: OrdersRepository | None = None,
    ) -> None:
        settings = get_agent_settings()
        self._llm = llm
        self._gatekeeper = gatekeeper
        self._registry = registry
        self._tools = tools
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
        self._tracer = tracer
        self._slos = slos
        self._grounding = grounding
        self._deliberation_markers = [m.casefold() for m in settings.deliberation_markers]
        self._max_rounds = settings.max_tool_iterations
        self._actor = settings.default_actor
        self._role = settings.default_role
        self._strategies = StrategySelector()
        # The window the model is handed is measured, not assumed: five
        # failure modes, checked on what was actually assembled.
        self._context_health = ContextFailureDetector()
        self._diagnoser = GroundingReasoningDiagnoser()
        self._intents = IntentClassifier()
        self._scope = ScopeGuard()
        self._authenticator = authenticator
        self._orders = orders
        # Who each session has proved itself to be. Authentication is per
        # conversation: a customer verifies once and stays verified for
        # that session, rather than retyping a code every turn.
        self._sessions: dict[str, Customer] = {}
        # The order list each session was last shown, so a bare "2"
        # resolves back to an order.
        self._menus: dict[str, OrderMenu] = {}
        self._history_turns = settings.history_turns
        self._conversations: dict[str, list[Message]] = {}
        self._last_intent: dict[str, Intent] = {}
        self._duplicates = DuplicateActionDetector()
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
        self._duplicates = DuplicateActionDetector()
        initial = AgentState(
            query=query,
            session_id=session_id,
            actor=self._actor,
            customer_id=customer.customer_id if customer else None,
        )
        final = await self._graph.ainvoke(initial)
        state = AgentState.model_validate(final)
        self._remember_turn(session_id, query, state.response)
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

    async def screen(self, state: AgentState) -> dict[str, Any]:
        """Is this the agent's business at all?

        First, and deliberately cheap: an off-topic message is refused by
        a regex pass rather than a completion. The refusal is identical
        every time, so paying a model to write it is pure waste.
        """
        self._report("scope_check")
        verdict = self._scope.check(state.query)
        self._done("scope_check", detail=verdict.topic)
        if verdict.in_scope:
            return {"decisions": [*state.decisions, f"scope: {verdict.topic}"]}
        return {
            "response": verdict.refusal,
            "decisions": [*state.decisions, "scope: refused (out of scope)"],
            "metadata": {**state.metadata, "halted": "out_of_scope"},
        }

    async def authenticate(self, state: AgentState) -> dict[str, Any]:
        """Establish who is speaking before looking anything up.

        Until a valid code arrives there is no owner to scope a lookup
        to, so "where is my order" has no answer. Verification is a
        database lookup, not a model call: the code either matches a row
        or it does not, and a deterministic check cannot be talked into a
        different answer.
        """
        existing = self._sessions.get(state.session_id) or current_customer()
        if existing is not None:
            # Already verified — either earlier in this conversation, or
            # by the caller before it handed the request over. A
            # deployment behind SSO authenticates upstream and binds the
            # customer; it should not be asked for a code it never issued.
            set_current_customer(existing)
            if state.session_id:
                self._sessions[state.session_id] = existing
            return {"customer_id": existing.customer_id}
        if self._authenticator is None:
            return {}
        self._report("authentication")
        result = self._authenticator.authenticate(state.query)
        self._done("authentication", detail="verified" if result.authenticated else "awaiting code")
        if not result.authenticated:
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

        Only a message that is *nothing but* a choice lands here.
        "refund order 2" states an intent and continues to the normal
        path, where the Gatekeeper and the refund rules apply.
        """
        menu = self._menus.get(state.session_id)
        if menu is None or self._orders is None or state.customer_id is None:
            return {}
        if _asks_for_the_list(state.query):
            orders = customer_orders(self._orders, state.customer_id)
            self._menus[state.session_id] = build_menu(orders)
            return self._halt(state, render_order_list(orders), "menu: listed")
        order_id = resolve_selection(state.query, menu)
        if order_id is None:
            return {}
        self._report("order_lookup")
        order = self._orders.get_for_customer(order_id, state.customer_id)
        self._done("order_lookup", detail=str(order_id))
        if order is None:
            return self._halt(
                state, f"I couldn't find order {order_id} on your account.", "menu: not found"
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
        self._memory.note(
            key=key,
            value=value,
            session_id=state.session_id,
            source=self._actor,
            category=category,
        )
        self._done("working_memory_write", detail=key)

    def _halt(self, state: AgentState, response: str, decision: str) -> dict[str, Any]:
        """Answer now, without reaching the model."""
        return {
            "response": response,
            "decisions": [*state.decisions, decision],
            "metadata": {**state.metadata, "halted": "answered_from_db"},
        }

    async def intake(self, state: AgentState) -> dict[str, Any]:
        """Screen the query before anything acts on it.

        Untrusted input is delimited and assessed here, at the edge —
        not after it has already been concatenated into a prompt where
        nothing downstream can tell instruction from data.
        """
        self._report("prompt_injection_check")
        assessment = self._security.prepare_untrusted(
            self._system_prompt(), state.query, self._registry.all_tools()
        )
        moderation = self._security.moderate(state.query)
        self._report("intent_classification")
        intent = await self._intents.classify_async(
            state.query, previous=self._last_intent.get(state.session_id)
        )
        if state.session_id:
            self._last_intent[state.session_id] = intent
        self._done("intent_classification", detail=intent.name)
        self._note(state, "current_intent", intent.name, category="intent")
        plan = self._deliberation.plan(state.query, intent.name, list(intent.allowed_tools))
        decisions = [
            f"intent: {intent.name}"
            + (f" (matched '{intent.matched_phrase}')" if intent.matched_phrase else ""),
            f"injection: {assessment.reason}",
        ]
        if moderation.flagged:
            decisions.append(f"moderation: {moderation.reason}")
        if self._memory is not None:
            self._memory.record_event(f"query: {state.query}", self._actor, state.session_id)
            decisions.extend(self._offer_to_memory(state))
        self._done("prompt_injection_check", detail=assessment.reason)
        if not plan.is_single_step:
            decisions.append(f"plan: {len(plan.steps)} steps — {plan.steps[0].description}")
        return {
            "decisions": [*state.decisions, *decisions],
            "metadata": {
                **state.metadata,
                "allowed_actions": assessment.allowed_actions,
                "intent": intent.name,
                "intent_tools": intent.allowed_tools,
                "plan": [step.model_dump() for step in plan.steps],
            },
        }

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
        result = self._memory.remember(
            state.query,
            source=f"customer:{state.customer_id}" if state.customer_id else "customer",
            provenance_kind="self_reported",
            session_id=state.session_id,
        )
        if result.route == "discard":
            return []
        self._done("memory_write", detail=result.route)
        if result.route == "refused":
            self._logger.warning(
                "Refused a memory write from conversation",
                reason=result.reason,
                session_id=state.session_id,
            )
        return [f"memory: {result.route} ({result.reason})"]

    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window: policy, recalled memory, and the query."""
        self._report("context_assembly")
        if self._context is None:
            return {"context": policy_text()}
        assembled = await self._context.assemble(
            Task(query=state.query, session_id=state.session_id, actor=state.actor)
        )
        fitted = self._strategies.fit(assembled)
        decisions = [f"context: {fitted.used_tokens}/{fitted.budget_tokens} tokens"]
        # Rot, distraction, confusion, clash, poisoning — measured on the
        # window that was actually built, because that is the only window
        # the model will see. A detector run over the candidate pool
        # reports on something nobody was given.
        modes = self._context_health.assess(fitted, set(tokenize(state.query)))
        if modes:
            self._logger.warning("Context window shows failure modes", modes=modes)
            decisions.append(f"context health: {', '.join(modes)}")
        if self._memory is not None and state.session_id:
            held = self._memory.working.size(state.session_id)
            self._done("memory_recall", detail=f"{held} fact(s) held for this conversation")
        self._done("context_assembly", detail=f"{fitted.used_tokens} tokens")
        return {"context": fitted.text, "decisions": [*state.decisions, *decisions]}

    async def reason(self, state: AgentState) -> dict[str, Any]:
        """One model turn, with the tools this turn is allowed to use."""
        self._report("llm_call")
        tools = await self._offered_tools(state)
        offered = sorted(tool.qualified_name for tool in tools)
        request = CompletionRequest(
            messages=self._prompt_messages(state),
            tools=self._registry.to_openai_tools([tool.qualified_name for tool in tools]),
        )
        result = await self._call_model(request, state, f"{state.session_id}:{state.tool_rounds}")
        message = Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
        self._done("llm_response_received", tokens=result.usage.total_tokens)
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
        except Exception:
            self._note_recovery(state, attempts_before, succeeded=False)
            self._done_failed("llm_call")
            raise
        self._note_recovery(state, attempts_before, succeeded=True)
        self._account_for(result, state)
        return result

    def _note_recovery(self, state: AgentState, attempts_before: int, succeeded: bool) -> None:
        """Record whether the resilience patterns actually recovered anything.

        Only a call that needed more than one attempt counts: a call that
        succeeded first time neither attempted nor achieved a recovery,
        and counting it as a success would report a recovery rate of
        100% for a system that has never had to recover.
        """
        if self._resilient.attempts - attempts_before <= 1:
            return
        metadata = state.metadata
        metadata["recoveries_attempted"] = int(metadata.get("recoveries_attempted", 0)) + 1
        if succeeded:
            metadata["recoveries_succeeded"] = int(metadata.get("recoveries_succeeded", 0)) + 1

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

    def _record_slo(self, metric: str, value: float) -> None:
        """Feed one observation to the SLO registry, when one is wired."""
        if self._slos is not None:
            self._slos.record(metric, value)

    async def act(self, state: AgentState) -> dict[str, Any]:
        """Execute the requested tool calls, each through the Gatekeeper."""
        self._report("tool_execution")
        last = state.messages[-1] if state.messages else Message()
        outputs: list[Message] = []
        results: list[str] = []
        decisions: list[str] = []
        offered = set(state.metadata.get("offered_actions", []))
        for requested in last.tool_calls or []:
            self._report("gatekeeper_evaluation", detail=requested.name)
            text, decision, stage = await self._invoke(
                requested.name, requested.arguments, state, offered
            )
            self._done(stage, detail=requested.name)
            results.append(text)
            decisions.append(decision)
            self._note(state, f"result_of_{requested.name}", text, category="tool_result")
            outputs.append(
                Message(role="tool", content=text, tool_call_id=requested.id or requested.name)
            )
        return {
            "messages": outputs,
            "tool_results": [*state.tool_results, *results],
            "decisions": [*state.decisions, *decisions],
            "escalated": state.escalated or any("escalate" in d for d in decisions),
        }

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """One tool-free call so the customer always gets an answer.

        Without this, hitting the tool-iteration cap returns whatever the
        last turn happened to contain — often nothing. A cap is a
        control, not an excuse to hand the customer an empty reply.
        """
        self._report("finalize")
        last = state.messages[-1] if state.messages else Message()
        critique = self._deliberation.reflect(last.content, state.tool_results)
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
            grounded = await self._check_grounding(last.content, state)
            self._done("finalize")
            return {"response": last.content, "decisions": [*state.decisions, *grounded]}
        # The reflector already logged what it found; saying it twice
        # here would double every rejection in the trace.
        result = await self._llm.complete(
            CompletionRequest(
                messages=[
                    *self._prompt_messages(state),
                    Message(role="user", content=build_reply_instruction()),
                ]
            )
        )
        reply = result.content if is_customer_ready(result.content) else _FALLBACK_REPLY
        grounding = await self._check_grounding(reply, state)
        decisions = [*state.decisions, *grounding]
        if not critique.acceptable:
            decisions.append(f"reflection: revised ({critique.summary})")
        # Scored by something independent of what wrote it. One bounded
        # pass: the loop revises once and reports, rather than iterating
        # towards a sentence nobody is waiting for.
        score = await self._deliberation.score(reply, state.tool_results)
        if state.tool_results:
            decisions.append(f"evaluation: {score:.2f} supported")
        self._done("finalize", tokens=result.usage.total_tokens)
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

    async def _check_grounding(self, reply: str, state: AgentState) -> list[str]:
        """Check the reply against what the tools actually returned.

        Advisory, not blocking. An unsupported claim is recorded and fed
        to the hallucination SLO; it does not withhold the answer,
        because a detector that can be wrong must not be the thing that
        decides whether a customer hears anything at all.
        """
        if self._grounding is None or not state.tool_results:
            return []
        self._report("hallucination_check")
        evidence = [Chunk(content=text, kind="tool_result") for text in state.tool_results]
        assessment = await self._grounding.assess(reply, evidence)
        state.metadata["hallucination_rate"] = assessment.hallucination_rate
        self._done("hallucination_check", detail=f"{assessment.hallucination_rate:.0%} unsupported")
        if assessment.clean:
            return []
        # One symptom, two causes, opposite fixes: evidence that was
        # never retrieved is a retrieval problem, evidence that was
        # retrieved and ignored is a reasoning problem. Routing every
        # hallucination to "improve retrieval" is why some are never
        # fixed.
        kinds = {self._diagnoser.diagnose(claim, evidence) for claim in assessment.unsupported}
        self._logger.warning(
            "Reply contains claims the tool results do not support",
            unsupported=assessment.unsupported,
            failure_kinds=sorted(kinds),
        )
        return [
            f"grounding: {len(assessment.unsupported)} unsupported claim(s) "
            f"({', '.join(sorted(kinds))} failure)"
        ]

    # ---- wiring ------------------------------------------------------

    def _build(self) -> Any:
        """Assemble the state graph."""
        graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
        graph.add_node("screen", self.screen)
        graph.add_node("authenticate", self.authenticate)
        graph.add_node("choose_order", self.choose_order)
        graph.add_node("intake", self.intake)
        graph.add_node("assemble_context", self.assemble_context)
        graph.add_node("reason", self.reason)
        graph.add_node("act", self.act)
        graph.add_node("finalize", self.finalize)

        graph.set_entry_point("screen")
        graph.add_conditional_edges(
            "screen", self._after_screen, {"authenticate": "authenticate", "done": END}
        )
        graph.add_conditional_edges(
            "authenticate",
            self._after_authenticate,
            {"choose_order": "choose_order", "done": END},
        )
        graph.add_conditional_edges(
            "choose_order", self._after_choose, {"intake": "intake", "done": END}
        )
        graph.add_edge("intake", "assemble_context")
        graph.add_edge("assemble_context", "reason")
        graph.add_conditional_edges(
            "reason", self._after_reason, {"act": "act", "finalize": "finalize"}
        )
        graph.add_conditional_edges(
            "act", self._after_act, {"reason": "reason", "finalize": "finalize"}
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    def _after_screen(self, state: AgentState) -> str:
        """Out-of-scope messages are answered and finished, with no model call."""
        return "done" if state.metadata.get("halted") else "authenticate"

    def _after_authenticate(self, state: AgentState) -> str:
        """An unverified turn, or a just-verified one, ends here."""
        return "done" if state.metadata.get("halted") else "choose_order"

    def _after_choose(self, state: AgentState) -> str:
        """A menu choice is answered from the database and finishes here."""
        return "done" if state.metadata.get("halted") else "intake"

    def _after_reason(self, state: AgentState) -> str:
        """Act on requested tools, or finish."""
        last = state.messages[-1] if state.messages else Message()
        return "act" if last.tool_calls else "finalize"

    def _after_act(self, state: AgentState) -> str:
        """Loop back to reason unless the round cap is spent."""
        return "reason" if state.tool_rounds < self._max_rounds else "finalize"

    async def _invoke(
        self, name: str, arguments: str, state: AgentState, offered: set[str]
    ) -> tuple[str, str, str]:
        """Validate, authorize and run one tool call.

        Returns the text the model sees, the decision line the operator
        sees, and the feed stage this ended in — the model is told *what*
        happened, the log records *why*, and the feed shows a suppressed
        repeat as a repeat rather than as work done.
        """
        call = self._registry.build_call(name, arguments)
        if call is None:
            return (
                f"Tool '{name}' is unavailable or its arguments were invalid.",
                f"{name}: invalid",
                "tool_refused",
            )
        call = call.model_copy(update={"params": self._security.minimize(call.params)})
        authority = AuthorityContext(
            actor=state.actor or self._actor,
            roles=[self._role],
            granted_by="user_task",
            session_id=state.session_id,
            # The Gatekeeper checks this and audits the attempt, so a tool
            # the model was never offered is refused on the record rather
            # than quietly here.
            offered_actions=sorted(offered),
        )
        repeated = self._duplicates.seen_before(call)
        if repeated is not None:
            return (
                self._duplicates.reject(call),
                f"{call.action}: suppressed (repeat)",
                "tool_suppressed",
            )
        decision = await self._gatekeeper.evaluate(call, authority)
        if not decision.allowed:
            if decision.question == "offered":
                self._logger.warning("Model called a tool it was not offered", tool=call.action)
                return (
                    f"'{call.action}' is not available for this request.",
                    f"{call.action}: refused (not offered)",
                    "tool_refused",
                )
            return (
                f"That action was not permitted: {decision.reason}",
                f"{call.action}: {decision.verdict} ({decision.question}) — {decision.reason}",
                "tool_refused",
            )
        result = await self._registry.invoke(call)
        if result.success:
            self._gatekeeper.commit(call, authority)
            self._duplicates.record(call, result.output)
        return (
            result.output if result.success else f"Tool failed: {result.error}",
            f"{call.action}: {'ok' if result.success else 'failed'}",
            "tool_execution" if result.success else "tool_refused",
        )

    async def _offered_tools(self, state: AgentState) -> list[Any]:
        """Which tools this turn may use.

        Two independent narrowings, and both matter: the injection guard
        removes what untrusted content must not reach, and retrieval
        removes what this task does not need. Neither substitutes for the
        Gatekeeper, which still rules on whatever is actually called.
        """
        by_intent = set(state.metadata.get("intent_tools", []))
        if not by_intent:
            # A message that raises no actionable intent gets no tools.
            # It is answered, not acted upon.
            return []
        allowed = set(state.metadata.get("allowed_actions", []))
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
        selected = await self._selector.select(state.query)
        narrowed = [tool for tool in selected if tool in tools]
        return narrowed or tools

    def _prompt_messages(self, state: AgentState) -> list[Message]:
        """System prompt, assembled context, the query, and the turns so far."""
        system = self._system_prompt()
        if state.context:
            system = f"{system}\n\n# Retrieved context\n{state.context}"
        return [
            Message(role="system", content=system),
            *self.history_for(state.session_id),
            Message(role="user", content=state.query),
            *state.messages,
        ]

    def _report(self, stage: str, detail: str | None = None) -> None:
        """Emit a stage-started event for the TUI, when one is listening."""
        if self._reporter is not None:
            self._reporter.started(stage, detail=detail)

    def _done_failed(self, stage: str, detail: str | None = None) -> None:
        """Emit a stage-failed event so the feed shows what broke."""
        if self._reporter is not None:
            self._reporter.failed(stage, detail=detail)

    def _done(self, stage: str, tokens: int | None = None, detail: str | None = None) -> None:
        """Emit a stage-completed event, so the feed can close the line out."""
        if self._reporter is not None:
            self._reporter.completed(stage, tokens_delta=tokens, detail=detail)


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
