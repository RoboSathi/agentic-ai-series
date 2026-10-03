"""The front door: what is checked before the model sees anything.

Each stage here is a feature that may be withheld, and a withheld stage is
one that was never built — not one that runs and reports finding nothing.
A request stopped here never becomes a conversation turn: it is
quarantined, and the audit trail says why.
"""

from typing import Any

from agent_platform.basic_sanity import BasicSanityRequest
from agent_platform.resilience.dead_letter_queue import DeadLetterReason
from agent_platform.security import MessageScreener
from customer_support_agent.basic_sanity import owned_order_id
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import (
    basic_sanity_fields,
)


def _outbound(state: AgentState) -> str:
    """The text a remote detector may be shown.

    Redacted when the pii stage ran, raw when it was withheld. Masking an
    email does not hide "ignore previous instructions", so detection is
    unaffected — what changes is whose data reaches the provider.
    """
    return state.model_query or state.query


class ScreeningNodes(NodeSupport):
    async def pii_filtering(self, state: AgentState) -> dict[str, Any]:
        """Take personal data out of the message before the model sees it.

        The filter already ran on three surfaces — traces, memory writes
        and tool arguments — but not on the one that leaves the building.
        The customer's own sentence went to the provider verbatim, so an
        email address or a card number typed into a support chat was sent
        to a third party and nothing in the harness had an opinion about
        it.

        Redaction is not destructive here. `state.query` keeps what the
        customer actually wrote, because ownership checks, order
        resolution and mandates all reason about it in-process and would
        be wrong about a masked order number. `state.model_query` is what
        the prompt is built from. The distinction is the point: personal
        data may be *used* locally and must not be *transmitted*.

        Never blocks. Finding a phone number in a support message is
        ordinary, not an attack — the stage that refuses things is
        `prompt_injection`, which runs alongside this one.
        """
        outcome = MessageScreener(self._security).redact(state.query, self._reporter)
        return {
            "model_query": outcome.text,
            "decisions": [
                *state.decisions,
                f"pii: {', '.join(outcome.categories)} redacted"
                if outcome.categories
                else "pii: none found",
            ],
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
        fields = basic_sanity_fields(result)

        # The order the customer named is now the one under discussion —
        # passed or refused. "Refund order 10823" refused because it has
        # shipped still makes 10823 the subject, and "cancel it" next turn
        # means 10823, not whichever order last passed. Only an order proved
        # to be theirs is recorded. Store only the identifier in working
        # memory; order data itself must still come from an ownership-scoped
        # repository/tool read on the path that needs it.
        order_id = owned_order_id(result)
        if order_id is not None:
            self._note(state, ORDER_UNDER_DISCUSSION, f"order {order_id}", category="order")

        if result.passed:
            deferred = result.reason_code == "ORDER_REFERENCE_NOT_PRESENT"
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
            if order_id is not None:
                metadata[ORDER_UNDER_DISCUSSION] = order_id
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
            "response": result.message or "I couldn't safely continue with that request.",
            "decisions": [
                *state.decisions,
                f"basic_sanity: blocked ({result.reason_code})",
            ],
            "metadata": {
                **state.metadata,
                "basic_sanity": result.model_dump(mode="json"),
                "halted": "basic_sanity",
                **({ORDER_UNDER_DISCUSSION: order_id} if order_id is not None else {}),
            },
        }

    def _quarantine_security_request(
        self,
        state: AgentState,
        *,
        reason: DeadLetterReason,
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
                    # The platform quarantines on behalf of a principal, not
                    # a customer — this agent's principal happens to be one.
                    principal_id=(
                        str(state.customer_id) if state.customer_id is not None else None
                    ),
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
        outcome = await MessageScreener(self._security).injection(
            _outbound(state),
            self._reporter,
            instructions=self._system_prompt(),
            tools=self._registry.all_tools(),
        )
        if outcome.blocked:
            return self._quarantine_security_request(
                state,
                reason="prompt_injection",
                injection_detected=True,
                injection_reason=outcome.reason,
            )
        return {
            "decisions": [*state.decisions, f"injection: {outcome.reason}"],
            "metadata": {
                **state.metadata,
                "allowed_actions": outcome.allowed_actions,
            },
        }

    async def content_moderation(self, state: AgentState) -> dict[str, Any]:
        """Screen the message against the content policy."""
        outcome = await MessageScreener(self._security).moderation(_outbound(state), self._reporter)
        if outcome.blocked:
            return self._quarantine_security_request(
                state,
                reason="content_moderation",
                moderation_flagged=True,
                moderation_reason=outcome.reason,
                moderation_categories=outcome.categories,
            )
        return {
            "decisions": [*state.decisions],
            "metadata": {**state.metadata},
        }
