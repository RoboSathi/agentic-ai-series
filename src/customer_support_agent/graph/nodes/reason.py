"""The model call, and everything that decides what it may see.

`reason` is one half of the **Tool-Use Loop** — the model proposes a typed
tool call, `act` executes it, the observation returns here, bounded by
`max_tool_iterations`. It is deliberately not ReAct: there is no
natural-language thought step, and the proposal is a validated tool call
rather than parsed prose.

The model is a remote dependency like any other, so the call goes through
a timeout, a retry under an idempotency key, and a circuit breaker.
"""

from typing import Any

from agentic_commons.llm.types import CompletionRequest, CompletionResult, Message
from agentic_commons.resilience import (
    external_api_error_fields,
)
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import FALLBACK_REPLY
from customer_support_agent.prompts.system_prompt import build_system_prompt


class ReasoningNodes(NodeSupport):
    async def reason(self, state: AgentState) -> dict[str, Any]:
        """One model turn, with the tools this turn is allowed to use."""
        self._loop_step = state.tool_rounds + 1
        tools = await self._offered_tools(state)
        offered = sorted(tool.qualified_name for tool in tools)
        request = CompletionRequest(
            messages=self._prompt_messages(state),
            tools=self._registry.to_openai_tools([tool.qualified_name for tool in tools]),
        )
        self._report_llm_request(request, purpose="tool_selection")
        try:
            result = await self._call_model(request, state, f"{state.session_id}:{state.tool_rounds}")
        except Exception as exc:
            error_fields = external_api_error_fields(exc)
            self._done_failed(
                "resilience",
                summary="LLM unavailable after retries",
                event_kind="resilience",
                fields={
                    "dependency": self._resilient.name,
                    **(error_fields or {}),
                    **({"reason": str(exc)} if not error_fields else {}),
                },
            )
            return {
                "response": FALLBACK_REPLY,
                "tool_rounds": state.tool_rounds + 1,
                "escalated": True,
                "decisions": [*state.decisions, "resilience: llm unavailable after retries"],
                "metadata": {**state.metadata, "external_dependency_failed": True},
            }
        message = Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
        state.working_memory.add_current_turn(message)
        self._report_llm_response(result, purpose="tool_selection")
        return {
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
                "llm_response_received",
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
            # Recovered is the pattern working; exhausted is a failure.
            report = self._done if succeeded else self._done_failed
            report(
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

    def _system_prompt(self) -> str:
        """Return the agent system prompt for security screening and LLM calls."""
        return build_system_prompt()

    def _prompt_messages(self, state: AgentState) -> list[Message]:
        """Build the small provider payload from working memory and context."""
        system = build_system_prompt()
        if state.context is not None and state.context.text:
            system = f"{system}\n\n# Retrieved context\n{state.context.text}"
        if state.metadata.get("intent_execution_route") == "llm_fallback":
            system = (
                f"{system}\n\n"
                "# Intent fallback\n"
                "The lightweight intent classifier could not resolve this turn. "
                "Use the current conversation and current-turn tool results to "
                "infer what the customer means. You may ask a clarifying question "
                "when context is still insufficient. Never claim that a refund, "
                "approval, queueing, cancellation, or other business action "
                "occurred unless you actually call the corresponding tool and "
                "receive a successful tool result.\n"
            )
        return [
            Message(role="system", content=system),
            Message(role="user", content=state.model_query or state.query),
            *self._current_messages(state),
        ]
