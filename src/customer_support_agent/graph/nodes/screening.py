"""The front door: what is checked before the model sees anything.

Each stage here is a feature that may be withheld, and a withheld stage is
one that was never built — not one that runs and reports finding nothing.
A request stopped here never becomes a conversation turn: it is
quarantined, and the audit trail says why.
"""

from typing import Any

from agentic_commons.basic_sanity import BasicSanityRequest
from agentic_commons.config import get_settings
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
        spans = self._security.detect_pii(state.query)
        redacted = self._security.redact(state.query) if spans else state.query
        categories = sorted({span.category for span in spans})
        self._report(
            "pii_filtering",
            event_kind="security",
            summary="Removing personal data before the model call",
        )
        self._done(
            "pii_filtering",
            event_kind="security",
            summary=(
                f"PII filtered: {', '.join(categories)}"
                if categories
                else "No personal data found in the message"
            ),
            fields={
                "found": len(spans),
                "categories": categories,
                # What the provider will actually receive. Shown because
                # a redaction nobody can see is a redaction nobody can
                # check — and getting it wrong in either direction
                # matters.
                "model_query": redacted,
                "mode": get_settings().security.pii.mode,
            },
        )
        return {
            "model_query": redacted,
            "decisions": [
                *state.decisions,
                f"pii: {', '.join(categories)} redacted" if categories else "pii: none found",
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
        self._report(
            "prompt_injection_check",
            event_kind="security",
            summary="Checking untrusted input for injection",
        )
        assessment = await self._security.prepare_untrusted_async(
            self._system_prompt(), _outbound(state), self._registry.all_tools()
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
        moderation = await self._security.moderate_async(_outbound(state))
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
