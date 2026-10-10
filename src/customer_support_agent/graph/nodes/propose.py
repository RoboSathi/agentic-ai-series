"""The model call, and everything that decides what it may see.

`propose` is one half of the **Tool-Use Loop** — the model proposes a typed
tool call, `execute` runs it, the observation returns here, bounded by
`max_tool_iterations`. It is deliberately not ReAct: there is no
natural-language thought step, and the proposal is a validated tool call
rather than parsed prose.

The model is a remote dependency like any other, so the call goes through
a timeout, a retry under an idempotency key, and a circuit breaker.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message
from agent_platform.observability import Span
from agent_platform.observability.model_observer import MODEL_STAGE as LLM_CALL_SPAN
from agent_platform.resilience import (
    external_api_error_fields,
)
from customer_support_agent.graph.loop_decision import after_propose_step
from customer_support_agent.graph.nodes._support import (
    PREFETCHED,
    TURN_OFFERED_TOOLS,
    NodeSupport,
)
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import FALLBACK_REPLY
from customer_support_agent.prompts.system_prompt import build_system_prompt


class ProposalNodes(NodeSupport):
    async def propose(self, state: AgentState) -> dict[str, Any]:
        """One model turn, with the tools this turn is allowed to use."""
        self._steps.begin(state.tool_rounds + 1)
        tools = await self._turn_offer(state)
        offered = sorted(tool.qualified_name for tool in tools)
        # Normally already drawn by the prefetch stage; a repeat in the same turn is a no-op.
        self._instructions.announce(
            self._activation(state),
            prefetched=list(state.metadata.get(PREFETCHED, [])),
            offered=[tool.model_name for tool in tools],
        )
        request = CompletionRequest(
            messages=self._prompt_messages(state),
            tools=self._registry.to_openai_tools([tool.qualified_name for tool in tools]),
        )
        # The model may ask for a tool or answer; the response card names which it did.
        purpose = get_settings().observability.llm_purposes.next_action
        self._report_llm_request(request, purpose=purpose)
        try:
            result = await self._call_model(request, state)
        except Exception as exc:
            error_fields = external_api_error_fields(exc)
            self._done_failed(
                "resilience",
                summary="LLM unavailable after retries",
                event_kind="resilience",
                fields={
                    "dependency": self._llm.model,
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
        self._report_llm_response(result, purpose=purpose)
        requested = bool(result.tool_calls)
        next_step, because = after_propose_step(requested)
        self._report_loop_status(
            tool_rounds=state.tool_rounds + 1,
            tools_requested=requested,
            invocations=list(state.metadata.get("last_tool_invocations") or []),
            escalated=state.escalated,
            refusal_final=bool(state.metadata.get("refusal_is_final")),
            next_step=next_step,
            because=because,
        )
        return {
            "tool_rounds": state.tool_rounds + 1,
            # The offered set is carried, not recomputed. Selection can
            # involve a network-backed ranker, so recomputing it in `execute`
            # could produce a different set and refuse a tool the model
            # was genuinely handed. An authority decision is made once.
            "metadata": {
                **state.metadata,
                "last_usage": result.usage.model_dump(),
                "offered_actions": offered,
            },
        }

    async def _call_model(self, request: CompletionRequest, state: AgentState) -> CompletionResult:
        """One model call, traced.

        Its protection — retry with backoff, circuit breaker, per-attempt timeout and a backup
        model — is the client's own (`create_llm_client` → `ModelChainBuilder`), so the node
        never wraps the model again: a second retry layer here would multiply the attempts.
        """
        span = self._open_span(state)
        try:
            result = await self._llm.complete(request)
        except Exception as exc:
            self._close_span(span, None, f"error: {exc!r}")
            error_fields = external_api_error_fields(exc)
            self._note_recovery(state, attempts=0, succeeded=False, error_fields=error_fields)
            self._done_failed(
                "llm_response_received",
                summary="LLM call failed",
                event_kind="model",
                fields=error_fields,
            )
            if error_fields:
                self._logger.error("LLM API call failed", **error_fields)
            raise
        self._note_recovery(state, attempts=result.attempts, succeeded=True)
        self._close_span(span, result, result.finish_reason or "")
        return result

    def _note_recovery(
        self,
        state: AgentState,
        attempts: int,
        succeeded: bool,
        error_fields: dict[str, Any] | None = None,
    ) -> None:
        """Record whether the resilience patterns actually recovered anything.

        Only a call that needed more than one attempt counts: a call that
        succeeded first time neither attempted nor achieved a recovery,
        and counting it as a success would report a recovery rate of
        100% for a system that has never had to recover.
        """
        # A failed call exhausted its retries (and its backups) by definition; a successful one
        # recovered only when it took more than one attempt.
        if succeeded and attempts <= 1:
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

    def _open_span(self, state: AgentState) -> Span | None:
        """Open the call's span before the call, so its latency is the call's.

        start/end rather than the context manager: those are the Protocol's
        methods, so a LangSmith-backed tracer works here without the graph
        knowing which one it was given.
        """
        if self._tracer is None:
            return None
        return self._tracer.start_span(LLM_CALL_SPAN, actor=state.actor)

    def _close_span(self, span: Span | None, result: CompletionResult | None, outcome: str) -> None:
        """Close the span with what the call used. Tokens, not money (`accounting.py`)."""
        if self._tracer is None or span is None:
            return
        if result is not None:
            span.model_version = result.model
            span.attributes.update(result.usage.model_dump())
        self._tracer.end_span(span, result=outcome)

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
            # The turn's procedures still narrow it: what they use is what the
            # task they were chosen for needs (ADR 0010).
            by_procedure = self._procedure_tools(state)
            return [
                tool
                for tool in self._registry.all_tools()
                if (not allowed or tool.qualified_name in allowed)
                and (not by_procedure or tool.name in by_procedure)
            ]

        by_intent = set(state.metadata.get("intent_tools", []))
        if not by_intent:
            return []
        by_procedure = self._procedure_tools(state)
        tools = [
            tool
            for tool in self._registry.all_tools()
            if tool.name in by_intent
            and (not by_procedure or tool.name in by_procedure)
            and (not allowed or tool.qualified_name in allowed)
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

        mandatory = [tool for tool in tools if tool.name in plan_tools]

        selected = await self._selector.select(state.query)
        narrowed = [tool for tool in selected if tool in tools]

        mandatory_names = {tool.qualified_name for tool in mandatory}
        return (
            mandatory + [tool for tool in narrowed if tool.qualified_name not in mandatory_names]
            or mandatory
            or tools
        )

    async def _turn_offer(self, state: AgentState) -> list[Any]:
        """The tools settled for this turn before its first read — the ones its card names."""
        settled = state.metadata.get(TURN_OFFERED_TOOLS)
        if settled is None:
            return await self._offered_tools(state)
        return [tool for name in settled if (tool := self._registry.get(name)) is not None]

    def _system_prompt(self) -> str:
        """The profile and policy, without per-turn procedures — what a screen sees."""
        return build_system_prompt(
            self._instructions.profile_section(), refund_limit=self._refund_limit()
        )

    def _prompt_messages(self, state: AgentState) -> list[Message]:
        """Build the small provider payload from working memory and context."""
        system = build_system_prompt(
            self._instructions.render(self._activation(state)), refund_limit=self._refund_limit()
        )
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
