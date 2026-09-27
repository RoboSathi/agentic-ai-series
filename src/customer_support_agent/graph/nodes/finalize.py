"""Turning what happened into something a customer can read.

The draft is critiqued once by the Output Quality Gate, checked against
the evidence the tools actually returned, and repaired at most once. A
claim no tool result supports does not reach the customer.
"""

from typing import Any

from agentic_commons.guardrails import CustomerResponse
from agentic_commons.hallucination_protection.checker import Resampler
from agentic_commons.llm.types import CompletionRequest, Message
from agentic_commons.resilience import (
    external_api_error_fields,
)
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
            evidence = self._grounding_evidence(state)
            guardrail_result = await self._response_guardrail.enforce(
                last.content, evidence, resample=self._resampler(state)
            )
            # No schema line here: this draft never went through the JSON
            # envelope, and reporting it as valid described a check that
            # did not run.
            self._report_grounding(
                blocked=guardrail_result.escalated,
                summary=_grounding_headline(guardrail_result),
                fields=guardrail_result.model_dump(exclude={"response"}),
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
                    summary=_grounding_headline(guardrail_result),
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
        request = CompletionRequest(
            messages=self._reply_messages(state),
            json_schema=CustomerResponse.model_json_schema(),
        )
        self._report_llm_request(request, purpose="reply_revision")
        try:
            result = await self._call_model(
                request,
                state,
                f"{state.session_id}:finalize",
            )
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
            repaired_result = await self._call_model(
                request,
                state,
                f"{state.session_id}:hallucination-repair",
            )
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
        guardrail_fields = guardrail_result.model_dump(
            exclude={"response"},
        )
        self._report_grounding(
            blocked=guardrail_result.escalated,
            summary=_grounding_headline(guardrail_result),
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
                result = await self._call_model(
                    request,
                    state,
                    key=f"{state.session_id}:resample:{index}",
                )
                self._report_llm_response(result, purpose=purpose)
                results.append(result)
            return [result.content for result in results if result.content]

        return resample

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

    def _is_deliberation(self, content: str) -> bool:
        """Whether this text is the model thinking rather than replying.

        A reply addresses the customer; deliberation addresses the model
        itself, or refers to the customer in the third person. Shipping
        the latter reads as "We need to respond as per the conversation",
        which is the model's notes, not an answer.
        """
        lowered = content.casefold()
        return any(marker in lowered for marker in self._deliberation_markers)


def _grounding_headline(result: Any) -> str:
    """The card title: the outcome when blocked, the claim tally when not."""
    if result.escalated:
        return str(result.outcome)
    return str(result.tally)
