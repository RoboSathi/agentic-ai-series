"""What the customer is asking for, and what that permits.

Classification does not authorize anything. It narrows which tools are
offered this turn; the Gatekeeper still rules on every side effect. A
read-shaped intent is answered here deterministically, with no model call
at all, which is why `read_intent` sits beside `classify_intent`.
"""

import json
from typing import Any

from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.intent.router import Intent


class IntentNodes(NodeSupport):
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
