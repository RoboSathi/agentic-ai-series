"""Turning what happened into something a customer can read.

The draft is critiqued once by the Output Quality Gate, checked against
the evidence the tools actually returned, and repaired at most once. A
claim no tool result supports does not reach the customer.
"""

from typing import Any

from agentic_commons.context import (
    Chunk,
)
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
)


class FinalizeNodes(NodeSupport):
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
                "response": FALLBACK_REPLY,
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
            guardrail_result = await self._response_guardrail.enforce(
                last.content, evidence, resample=self._resampler(state)
            )
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
                "response": FALLBACK_REPLY,
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
            return repaired or FALLBACK_REPLY

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

    def _resampler(self, state: AgentState) -> Resampler:
        """Ask the model the same question again, n more times.

        Handed to the detection ladder, which calls it only where the
        entailment tier could not decide — so on an ordinary turn this
        closure is created and never invoked.
        """

        async def resample(count: int) -> list[str]:
            request = CompletionRequest(messages=self._prompt_messages(state))
            results = [
                await self._call_model(request, state, key=f"{state.session_id}:resample:{index}")
                for index in range(max(0, count))
            ]
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
