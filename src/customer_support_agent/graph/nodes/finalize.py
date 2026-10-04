"""Turning what happened into something a customer can read.

The draft is critiqued once by the Output Quality Gate, checked against
the evidence the tools actually returned, and repaired at most once. A
claim no tool result supports does not reach the customer.
"""

from typing import Any

from agent_platform.guardrails import CustomerResponse, ResponseGuardrailResult
from agent_platform.hallucination_protection.checker import Resampler
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.resilience import (
    external_api_error_fields,
)
from customer_support_agent.config import get_agent_settings
from customer_support_agent.control_loop.quality_gate import Critique
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import (
    FALLBACK_REPLY,
    is_customer_ready,
)
from customer_support_agent.prompts.system_prompt import (
    build_reply_instruction,
    build_reply_system_prompt,
)

# What `finalize` does with the draft, as the hallucination check's pre-check line states it.
_SEND_DRAFT = "send_draft"
_REVISE_REPLY = "revise_reply"
_GUARD_WITHOUT_EVIDENCE = "guard_without_evidence"
_PASSED = "passed"
_FAILED = "failed"
_NO_EVIDENCE = "none"
# Where the draft check's line waits for the hallucination check card.
_PRE_CHECK = "pre_check"


class FinalizeNodes(NodeSupport):
    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """One tool-free call so the customer always gets an answer.

        Without this, hitting the tool-iteration cap returns whatever the
        last turn happened to contain — often nothing. A cap is a
        control, not an excuse to hand the customer an empty reply.
        """
        # The loop is over; the reply is composed outside it.
        self._loop_step = None
        self._report(
            "finalize",
            summary="Preparing the customer-facing response",
        )
        # A previous model call can already have exhausted its retry budget.
        # Do not immediately spend another retry budget on finalize; surface
        # the outage and hand the turn to the human-facing fallback.
        if state.metadata.get("external_dependency_failed"):
            return {
                "response": FALLBACK_REPLY,
                "decisions": [*state.decisions, "resilience: llm unavailable after retries"],
                "escalated": True,
                "metadata": state.metadata,
            }
        messages = self._current_messages(state)
        last = messages[-1] if messages else Message()
        authoritative_evidence = self._authoritative_observations(state)
        critique = self._deliberation.reflect(last.content, authoritative_evidence)
        sendable = (
            last.role == "assistant"
            and not last.tool_calls
            and is_customer_ready(last.content)
            and not self._is_deliberation(last.content)
            # The draft check decides whether the draft in hand is good
            # enough to send, or whether the second call is worth its cost.
            # Without it the only question asked was "is this shaped like
            # a reply", which a confident wrong answer also passes.
            and critique.acceptable
        )
        if sendable:
            next_step = _SEND_DRAFT
        elif not critique.acceptable and not authoritative_evidence:
            next_step = _GUARD_WITHOUT_EVIDENCE
        else:
            next_step = _REVISE_REPLY
        self._report_draft_check(state, last.content, critique, next_step)
        if sendable:
            # Grounding runs on this path too. It used to be checked only
            # when a second call was needed, which is the *rarer* path —
            # so the reply that usually reached the customer was the one
            # never checked.
            evidence = self._grounding_evidence(state)
            guardrail_result = await self._response_guardrail.enforce(
                last.content, evidence, resample=self._resampler(state)
            )
            # No schema line here: this draft never went through the JSON
            # envelope, and reporting it as valid described a check that
            # did not run.
            self._report_grounding(state, guardrail_result)
            return {
                "response": guardrail_result.response,
                "decisions": [
                    *state.decisions,
                    f"grounding: {guardrail_result.reason_code}",
                ],
                "escalated": state.escalated or guardrail_result.escalated,
            }
        # If the draft check found an issue and there is no authoritative tool
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
                self._report_grounding(state, guardrail_result)
                return {
                    "response": guardrail_result.response,
                    "decisions": [
                        *state.decisions,
                        f"draft_check: failed ({critique.summary})",
                        f"grounding: {guardrail_result.reason_code}",
                    ],
                    "escalated": state.escalated or guardrail_result.escalated,
                }

        # The draft check already logged what it found; saying it twice
        # here would double every rejection in the trace.
        request = CompletionRequest(
            messages=self._reply_messages(state),
            json_schema=CustomerResponse.model_json_schema(),
        )
        self._report_llm_request(request, purpose="reply_revision")
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
                "decisions": [*state.decisions, "resilience: llm unavailable after retries"],
                "escalated": True,
                "metadata": {**state.metadata, "external_dependency_failed": True},
            }
        self._report_llm_response(
            result,
            purpose="reply_revision",
        )
        reply, schema_fields = self._response_guardrail.validate_schema(result.content)
        if reply is None or not is_customer_ready(reply):
            reply = FALLBACK_REPLY
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
                "strictly supported by the evidence below. Only a tool result can "
                "show that something happened; policy and knowledge support "
                "general statements. Remove any unsupported detail. Do not invent "
                "facts. Return only JSON matching the response schema.\n\n"
                f"Draft:\n{candidate}\n\n"
                "Evidence:\n"
                + "\n".join(
                    f"- [{chunk.kind}] {chunk.content}" for chunk in self._grounding_evidence(state)
                )
            )
            request = CompletionRequest(
                messages=[Message(role="user", content=repair_prompt)],
                json_schema=CustomerResponse.model_json_schema(),
            )
            self._report_llm_request(request, purpose="hallucination_repair")
            repaired_result = await self._call_model(request, state)
            self._report_llm_response(repaired_result, purpose="hallucination_repair")
            repaired, _ = self._response_guardrail.validate_schema(repaired_result.content)
            return repaired or FALLBACK_REPLY

        evidence = self._grounding_evidence(state)
        guardrail_result = await self._response_guardrail.enforce(
            reply,
            evidence,
            revise=revise_unsupported,
        )
        reply = guardrail_result.response
        self._report_grounding(state, guardrail_result)
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
            decisions.append(f"draft_check: revised ({critique.summary})")
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
                # Whether the revision came back as the JSON envelope the
                # schema asks for, or as plain text used as-is. The
                # grounding verdict has its own card and is not repeated.
                "schema_valid": schema_fields.get("schema_valid", True),
                "validation_mode": schema_fields.get("validation_mode"),
            },
        )
        return {
            "response": reply,
            "decisions": decisions,
        }

    def _report_draft_check(
        self, state: AgentState, draft: str, critique: Critique, next_step: str
    ) -> None:
        """Record the draft check for the hallucination check card, and log all of it.

        The draft check is the grounding guard's deterministic pre-filter,
        not an LLM critiquing itself: fixed rules (an empty draft, a
        deliberation, an order id no record holds) that decide only how the
        reply reaches the guard. So it is shown on the guard's card, as one
        line, and logged in full here.
        """
        problems = list(critique.problems)
        sources = self._evidence_sources(state)
        record = self._order_record(state)
        self._logger.info(
            "Draft check completed" if critique.acceptable else "Draft check failed",
            draft_check=_PASSED if critique.acceptable else _FAILED,
            problems=problems,
            next_step=next_step,
            checks_run=list(critique.checks),
            evidence_sources=sources,
            order_record=record.content if record is not None else "",
            draft=draft,
        )
        line = f"{_PASSED if critique.acceptable else _FAILED} → {next_step}"
        if problems:
            line += f" · {'; '.join(problems)}"
        line += f" · evidence: {', '.join(sources) or _NO_EVIDENCE}"
        state.metadata[_PRE_CHECK] = line

    def _reply_messages(self, state: AgentState) -> list[Message]:
        """The smallest payload that can write this reply.

        What writing a reply needs is what the customer asked and what
        the tools actually returned — the latter because the reply is
        checked against it immediately afterwards, so anything not in
        here is an unsupported claim by construction. What it does not
        need is the tool-selection prompt, the refund policy or the
        retrieved order corpus: no tool is offered on this call, there is
        nothing left to decide, and every one of those was re-read in
        full to produce two sentences.
        """
        return [
            Message(role="system", content=build_reply_system_prompt()),
            Message(role="user", content=state.model_query or state.query),
            *self._current_messages(state),
            Message(role="user", content=build_reply_instruction()),
        ]

    def _resampler(self, state: AgentState) -> Resampler:
        """Ask the model the same question again, n more times.

        Handed to the detection ladder, which calls it only where the
        entailment tier could not decide — so on an ordinary turn this
        closure is created and never invoked.
        """

        async def resample(count: int) -> list[str]:
            request = CompletionRequest(messages=self._prompt_messages(state))
            results = []
            for index in range(max(0, count)):
                purpose = f"response_resample_{index + 1}"
                self._report_llm_request(request, purpose=purpose)
                result = await self._call_model(request, state)
                self._report_llm_response(result, purpose=purpose)
                results.append(result)
            return [result.content for result in results if result.content]

        return resample

    def _report_grounding(self, state: AgentState, result: ResponseGuardrailResult) -> None:
        """Narrate the grounding stage — but only when there is one — and queue what it cut.

        With `hallucination_guard` withheld no detector is built, and
        `ResponseGuardrail.enforce` returns the draft unexamined. Printing
        "Grounded: 0/0 claims supported" for that would describe a check
        that never ran, which is worse than silence: it is the screen
        telling an operator a control is there when it is not. A stage
        that was not built has nothing to report.

        The card is coloured by what the customer received: the reply as
        written (done), the reply less a sentence (a warning), or nothing
        of it (blocked). A cut reply is queued for a person either way.
        """
        if self._grounding is None:
            return
        review = self._refer_reply(state, result)
        fields = result.model_dump(exclude={"response"})
        pre_check = state.metadata.get(_PRE_CHECK)
        if pre_check:
            fields[_PRE_CHECK] = pre_check
        if review is not None:
            fields["review_request"] = review
        summary = _grounding_headline(result, review)
        report = (
            self._reporter_blocked
            if result.withheld
            else self._reporter_warning
            if result.escalated
            else self._done
        )
        report("hallucination_check", summary=summary, event_kind="grounding", fields=fields)

    def _refer_reply(self, state: AgentState, result: ResponseGuardrailResult) -> int | None:
        """The review request a cut reply was queued as, when there is a queue to put it in."""
        if self._reply_referral is None:
            return None
        return self._reply_referral.refer(
            result, customer_id=state.customer_id, session_id=state.session_id
        )

    def _is_deliberation(self, content: str) -> bool:
        """Whether this text is the model thinking rather than replying.

        A reply addresses the customer; deliberation addresses the model
        itself, or refers to the customer in the third person. Shipping
        the latter reads as "We need to respond as per the conversation",
        which is the model's notes, not an answer.
        """
        lowered = content.casefold()
        return any(marker in lowered for marker in self._deliberation_markers)


def _grounding_headline(result: ResponseGuardrailResult, review: int | None) -> str:
    """The card title: what was sent when a claim was cut, the claim tally when not."""
    if not result.escalated:
        return str(result.tally)
    if review is None:
        return str(result.outcome)
    queued = get_agent_settings().review_output.queued.format(id=review)
    return f"{result.outcome} · {queued}"
