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
from typing import Any

from langgraph.graph import END, StateGraph

from agentic_commons.config import get_settings
from agentic_commons.context import FiveStagePipeline, StrategySelector, Task
from agentic_commons.evals import NollmScoringCascadeImpl
from agentic_commons.gatekeeper import AuthorityContext, GatekeeperImpl
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import CompletionRequest, Message
from agentic_commons.logging import get_logger
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability import StageReporter
from agentic_commons.security import SecuritySuite
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from customer_support_agent.control_loop import DuplicateActionDetector
from customer_support_agent.core.intent import Intent, IntentClassifier
from customer_support_agent.core.schema import AgentState, policy_text
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
    ) -> None:
        settings = get_settings().customer_support_agent
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
        self._max_rounds = settings.max_tool_iterations
        self._actor = settings.default_actor
        self._role = settings.default_role
        self._strategies = StrategySelector()
        self._intents = IntentClassifier()
        self._history_turns = settings.history_turns
        self._conversations: dict[str, list[Message]] = {}
        self._last_intent: dict[str, Intent] = {}
        self._duplicates = DuplicateActionDetector()
        self._logger = get_logger("customer_support_agent")
        self._graph = self._build()

    async def handle(self, query: str, session_id: str = "") -> AgentState:
        """Run one customer query end to end, in its conversation.

        Prior turns are replayed so a follow-up has something to refer
        to. Without them "what about 10010" is unanswerable: the subject
        lives in the previous turn, not in the words the customer just
        typed.
        """
        self._duplicates = DuplicateActionDetector()
        initial = AgentState(query=query, session_id=session_id, actor=self._actor)
        final = await self._graph.ainvoke(initial)
        state = AgentState.model_validate(final)
        self._remember_turn(session_id, query, state.response)
        return state

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

    async def intake(self, state: AgentState) -> dict[str, Any]:
        """Screen the query before anything acts on it.

        Untrusted input is delimited and assessed here, at the edge —
        not after it has already been concatenated into a prompt where
        nothing downstream can tell instruction from data.
        """
        self._report("prompt_injection_check")
        assessment = self._security.prepare_untrusted(
            build_system_prompt(), state.query, self._registry.all_tools()
        )
        moderation = self._security.moderate(state.query)
        self._report("intent_classified")
        intent = self._intents.classify(
            state.query, previous=self._last_intent.get(state.session_id)
        )
        if state.session_id:
            self._last_intent[state.session_id] = intent
        self._done("intent_classified", detail=intent.name)
        decisions = [
            f"intent: {intent.name}"
            + (f" (matched '{intent.matched_phrase}')" if intent.matched_phrase else ""),
            f"injection: {assessment.reason}",
        ]
        if moderation.flagged:
            decisions.append(f"moderation: {moderation.reason}")
        if self._memory is not None:
            self._memory.record_event(f"query: {state.query}", self._actor, state.session_id)
        self._done("prompt_injection_check", detail=assessment.reason)
        return {
            "decisions": [*state.decisions, *decisions],
            "metadata": {
                **state.metadata,
                "allowed_actions": assessment.allowed_actions,
                "intent": intent.name,
                "intent_tools": intent.allowed_tools,
            },
        }

    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window: policy, recalled memory, and the query."""
        self._report("context_assembly")
        if self._context is None:
            return {"context": policy_text()}
        assembled = await self._context.assemble(
            Task(query=state.query, session_id=state.session_id, actor=state.actor)
        )
        fitted = self._strategies.fit(assembled)
        self._done("context_assembly", detail=f"{fitted.used_tokens} tokens")
        return {
            "context": fitted.text,
            "decisions": [
                *state.decisions,
                f"context: {fitted.used_tokens}/{fitted.budget_tokens} tokens",
            ],
        }

    async def reason(self, state: AgentState) -> dict[str, Any]:
        """One model turn, with the tools this turn is allowed to use."""
        self._report("llm_call")
        tools = await self._offered_tools(state)
        result = await self._llm.complete(
            CompletionRequest(
                messages=self._prompt_messages(state),
                tools=self._registry.to_openai_tools([tool.qualified_name for tool in tools]),
            )
        )
        message = Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
        self._done("llm_response_received", tokens=result.usage.total_tokens)
        return {
            "messages": [message],
            "tool_rounds": state.tool_rounds + 1,
            "metadata": {**state.metadata, "last_usage": result.usage.model_dump()},
        }

    async def act(self, state: AgentState) -> dict[str, Any]:
        """Execute the requested tool calls, each through the Gatekeeper."""
        self._report("tool_execution")
        last = state.messages[-1] if state.messages else Message()
        outputs: list[Message] = []
        results: list[str] = []
        decisions: list[str] = []
        offered = {tool.qualified_name for tool in await self._offered_tools(state)}
        for requested in last.tool_calls or []:
            self._report("gatekeeper_evaluation", detail=requested.name)
            text, decision, stage = await self._invoke(
                requested.name, requested.arguments, state, offered
            )
            self._done(stage, detail=requested.name)
            results.append(text)
            decisions.append(decision)
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
        if last.role == "assistant" and is_customer_ready(last.content) and not last.tool_calls:
            self._done("finalize")
            return {"response": last.content}
        result = await self._llm.complete(
            CompletionRequest(
                messages=[
                    *self._prompt_messages(state),
                    Message(role="user", content=build_reply_instruction()),
                ]
            )
        )
        reply = result.content if is_customer_ready(result.content) else _FALLBACK_REPLY
        self._done("finalize", tokens=result.usage.total_tokens)
        return {
            "response": reply,
            "messages": [Message(role="assistant", content=reply)],
        }

    # ---- wiring ------------------------------------------------------

    def _build(self) -> Any:
        """Assemble the state graph."""
        graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
        graph.add_node("intake", self.intake)
        graph.add_node("assemble_context", self.assemble_context)
        graph.add_node("reason", self.reason)
        graph.add_node("act", self.act)
        graph.add_node("finalize", self.finalize)

        graph.set_entry_point("intake")
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
        system = build_system_prompt()
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

    def _done(self, stage: str, tokens: int | None = None, detail: str | None = None) -> None:
        """Emit a stage-completed event, so the feed can close the line out."""
        if self._reporter is not None:
            self._reporter.completed(stage, tokens_delta=tokens, detail=detail)


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
