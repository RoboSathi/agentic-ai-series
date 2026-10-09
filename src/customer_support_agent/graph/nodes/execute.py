"""Executing what the model proposed — the other half of the Tool-Use Loop.

Every proposal passes the loop control, then the Gatekeeper, then the
executor; and only a result that actually succeeded becomes an
authoritative fact the reply may be grounded on.

The order is the platform's single tool path (`agent_platform.tools.path`),
which also records where each proposal stopped. With the Gatekeeper withheld
the same refund that was escalated a moment ago simply happens: only the
authority decision is gone — the call still runs through the Executor, the
audit entry is still written (its standing is UNGOVERNED), and no budget is
committed, because no budget was consulted.
"""

from typing import Any

from agent_platform.audit import AuditEntry
from agent_platform.executor import (
    ApprovedToolCall,
    ExecutionOutcome,
)
from agent_platform.gatekeeper import AuthorityContext, AuthorityDecision
from agent_platform.gatekeeper.mandate import MandateSet
from agent_platform.llm.types import Message, ModelToolCall
from agent_platform.logging.trace_context import get_current_trace_id
from agent_platform.observability.types import utc_now
from agent_platform.observability.units import close_task, current_task_kind, open_task
from agent_platform.tools.path import DUPLICATE, NOT_OFFERED, ToolOutcome, ToolTurn
from agent_platform.tools.types import ToolCall, ToolResult
from agent_platform.verification import VerificationReport
from customer_support_agent.config.settings import get_agent_settings
from customer_support_agent.graph.loop_decision import ITERATION_LIMIT, after_execute_step
from customer_support_agent.graph.nodes._support import (
    TOOL_BLOCKED,
    TOOL_COMPLETED,
    TOOL_REFERRED,
    NodeSupport,
)
from customer_support_agent.graph.nodes.review_wait import AWAITING_REVIEW
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import (
    gatekeeper_fields,
    loop_control_fields,
    order_fields,
    tool_result_fields,
)

# Appended to every deterministic refusal. The model is good at reading
# instructions and bad at guessing whether an error is worth retrying;
# the duplicate-suppression path already learned this lesson.
_FINAL = (
    "This decision is deterministic and will not change if you call it again — "
    "do not retry it. Tell the customer what happened instead."
)

_ESCALATE = "escalate"

# A tool's status beyond the shared ones: it ran and failed, or ran and did not read back.
TOOL_FAILED, TOOL_UNVERIFIED = "failed", "unverified"

# Verdicts that cannot become an allow by re-proposing with different
# arguments. `escalate` is already terminal for the turn; these are the
# denials that are terminal for the same reason.
_TERMINAL_DENIALS = frozenset(
    {
        "TARGET_NOT_IN_SCOPE",
        "ACTION_NOT_OFFERED",
        "ACTOR_NOT_AUTHORIZED",
    }
)


def _has_terminal_denial(invocations: list[dict[str, Any]]) -> bool:
    """Whether any proposal this turn was refused for a reason that will not move."""
    return any(
        isinstance(item.get("gatekeeper"), dict)
        and item["gatekeeper"].get("decision") == "deny"
        and item["gatekeeper"].get("reason_code") in _TERMINAL_DENIALS
        for item in invocations
    )


def _queued_review(invocations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The refund this turn newly queued for a human, if it queued one.

    Only a *new* request: one already waiting belongs to the run that
    queued it, and that run is the one its decision resumes.
    """
    for item in invocations:
        result = item.get("tool_result")
        if isinstance(result, dict) and result.get("queued") and result.get("request_id"):
            return {
                "request_id": result["request_id"],
                "order_id": result.get("order_id"),
                "amount": result.get("amount"),
                # When the wait began, so the review's wait time can be recorded.
                "queued_at": utc_now().isoformat(),
            }
    return None


class ExecutionNodes(NodeSupport):
    async def execute(self, state: AgentState) -> dict[str, Any]:
        """Execute the requested tool calls, each through the Gatekeeper."""
        # The same iteration `propose` opened (it has already counted it).
        self._steps.begin(max(state.tool_rounds, 1))
        messages = self._current_messages(state)
        last = messages[-1] if messages else Message()
        requests = last.tool_calls or []
        decisions: list[str] = []
        invocation_fields_history: list[dict[str, Any]] = []
        offered = set(state.metadata.get("offered_actions", []))
        for requested in requests:
            self._steps.tool_request(
                requested.name, requested.arguments, redact=self._security.redact
            )
            text, decision, stage, invocation_fields = await self._invoke(
                requested.name, requested.arguments, state, offered
            )
            invocation_fields_history.append(invocation_fields)
            decisions.append(decision)
            tool_message = Message(
                role="tool", content=text, tool_call_id=requested.id or requested.name
            )
            self._add_action(
                state,
                requested.name,
                requested.arguments,
                str(invocation_fields.get("evidence", text)),
                outcome=str(invocation_fields.get("tool_status", TOOL_BLOCKED)),
            )
            state.working_memory.add_current_turn(tool_message)

            # Once a proposal escalates, autonomous execution stops for the
            # remainder of this assistant turn. Do not execute later batched
            # side effects from the same model message.
            gatekeeper = invocation_fields.get("gatekeeper")
            if isinstance(gatekeeper, dict) and gatekeeper.get("decision") == _ESCALATE:
                remaining = len(requests) - len(invocation_fields_history)
                if remaining:
                    self._reporter_warning(
                        "gatekeeper_evaluation",
                        summary=f"Skipped {remaining} tool proposal(s) after escalation",
                        event_kind="gatekeeper",
                        fields={
                            "decision": _ESCALATE,
                            "reason_code": "AUTONOMY_STOPPED_AFTER_ESCALATION",
                            "skipped_requests": remaining,
                        },
                    )
                break

        turn_escalated = any(
            item.get("gatekeeper", {}).get("decision") == _ESCALATE
            for item in invocation_fields_history
            if isinstance(item.get("gatekeeper"), dict)
        )
        self._report_after_act(
            state,
            invocation_fields_history,
            escalated=state.escalated or turn_escalated,
            refusal_final=_has_terminal_denial(invocation_fields_history),
        )
        return {
            "decisions": decisions,
            "escalated": state.escalated or turn_escalated,
            "metadata": {
                **state.metadata,
                AWAITING_REVIEW: _queued_review(invocation_fields_history)
                or state.metadata.get(AWAITING_REVIEW),
                "last_tool_invocations": invocation_fields_history,
                # A refusal that cannot become an allow ends the tool loop
                # the way an escalation does. Sending the model back to
                # reason() after one bought three identical proposals and
                # the iteration ceiling.
                "refusal_is_final": _has_terminal_denial(invocation_fields_history),
            },
        }

    def _report_after_act(
        self,
        state: AgentState,
        invocations: list[dict[str, Any]],
        *,
        escalated: bool,
        refusal_final: bool,
    ) -> None:
        """Report where the loop goes after this round, as the graph will route it."""
        next_step, because = after_execute_step(
            state.tool_rounds, self._max_rounds, escalated, refusal_final
        )
        self._report_loop_status(
            tool_rounds=state.tool_rounds,
            tools_requested=True,
            invocations=invocations,
            escalated=escalated,
            refusal_final=refusal_final,
            next_step=next_step,
            because=because,
        )
        if because == ITERATION_LIMIT:
            report = self._loop_control.evaluate_round(state.tool_rounds + 1, self._max_rounds)
            self._reporter_blocked(
                "loop_control",
                summary="Maximum tool iterations reached",
                event_kind="loop_control",
                fields=loop_control_fields(report),
            )

    async def _invoke(
        self, name: str, arguments: str, state: AgentState, offered: set[str]
    ) -> tuple[str, str, str, dict[str, Any]]:
        """One tool call through the platform's single tool path, drawn as this agent draws it.

        The path validates, suppresses duplicates, asks the Gatekeeper (or runs
        UNGOVERNED, audited, when it is withheld) and executes; this method only
        turns where the call stopped into the cards and the text the model reads.
        """
        outcome = await self._tool_path.run(
            ModelToolCall(name=name, arguments=arguments),
            ToolTurn(
                loop=self._loop_control,
                round_number=state.tool_rounds,
                max_rounds=self._max_rounds,
                authority=lambda call: self._authority(call, state, offered),
                prepare=lambda call: call.model_copy(
                    update={"params": self._security.minimize(call.params)}
                ),
                on_authorizing=self._authorizing,
                on_decision=self._decided,
            ),
        )
        result = await self._described(name, outcome)
        self._settle_task(outcome, result[3])
        return result

    async def _described(
        self, name: str, outcome: ToolOutcome
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Where the call stopped, as cards, model text and invocation fields."""
        if outcome.call is None:
            return self._rejected(name, outcome)
        control_fields = loop_control_fields(outcome.control) if outcome.control else {}
        if outcome.decision is None and outcome.execution is None:
            return self._suppressed(outcome.call, outcome, control_fields)
        if outcome.decision is not None and not outcome.decision.allowed:
            return await self._refused(outcome, control_fields)
        decision_fields = (
            gatekeeper_fields(outcome.decision, outcome.call) if outcome.decision else {}
        )
        return await self._after_execution(
            outcome.call, outcome.execution, control_fields, decision_fields
        )

    @staticmethod
    def _settle_task(outcome: ToolOutcome, fields: dict[str, Any]) -> None:
        """A write ends the task it carries out: issued, queued for review, refused or failed.

        A write the model made with no task open (the intent was unclear) opens
        the task its action belongs to first, so the outcome is never lost to
        an implicit task. A refusal the model may still recover from by asking
        differently leaves the task open.
        """
        call = outcome.call
        if call is None or not call.side_effecting:
            return
        words = get_agent_settings().tasks
        kind = next((k for k, spec in words.kinds.items() if spec.action == call.action), None)
        if kind is None:
            return
        status = fields.get("tool_status")
        ending = {
            TOOL_COMPLETED: words.kinds[kind].done,
            TOOL_REFERRED: words.queued,
            TOOL_FAILED: words.failed,
            TOOL_UNVERIFIED: words.unverified,
        }.get(str(status))
        if ending is None and _has_terminal_denial([fields]):
            ending = words.refused
        if ending is None:
            return
        if current_task_kind() != kind:
            open_task(kind)
        close_task(ending)

    def _authority(self, call: ToolCall, state: AgentState, offered: set[str]) -> AuthorityContext:
        """Who is asking, what they were offered, and what they actually asked for."""
        return AuthorityContext(
            actor=state.actor or self._actor,
            granted_by="user_task",
            session_id=state.session_id,
            offered_actions=sorted(offered),
            request_context={
                "query": state.query,
                "customer_id": state.customer_id,
                "metadata": dict(state.metadata),
                "tool_action": call.action,
                "tool_params": dict(call.params),
            },
            # What this customer actually asked for, this turn. Issued
            # from the request and the record before the model was
            # consulted, so a proposal is checked against the ask rather
            # than against its own plausibility. Only a write is scoped, so
            # only a write has mandates issued for it.
            mandates=self._mandates_for(state) if call.side_effecting else None,
            turn=state.tool_rounds,
        )

    def _authorizing(self, call: ToolCall) -> None:
        self._report(
            "gatekeeper_evaluation",
            event_kind="gatekeeper",
            summary="Checking deterministic execution admissibility",
            fields={"requested_action": call.action},
        )

    def _decided(self, call: ToolCall, decision: AuthorityDecision) -> None:
        """The Gatekeeper's card: an allow, a caution (escalate) or a refusal."""
        # An escalation is the harness working as designed — a person takes
        # over — so the platform card reads it as a caution, not a refusal.
        self._steps.gatekeeper(call, decision, extra=order_fields(call))

    def _rejected(self, name: str, outcome: ToolOutcome) -> tuple[str, str, str, dict[str, Any]]:
        """An unknown tool, or arguments its schema refused."""
        blocked_fields = {
            "tool": name,
            "result": "invalid_request",
            "reason_code": "TOOL_CALL_INVALID",
            "reason": outcome.reason or f"tool is {outcome.stage}",
            "stage": outcome.stage,
        }
        self._reporter_blocked(
            "tool_result",
            summary=f"Tool request rejected: {name}",
            event_kind="tool_result",
            fields=blocked_fields,
        )
        return (
            f"Tool '{name}' is unavailable or its arguments were invalid.",
            f"{name}: invalid",
            "tool_refused",
            {"tool_result": blocked_fields, "tool_status": TOOL_BLOCKED},
        )

    def _suppressed(
        self, call: ToolCall, outcome: ToolOutcome, control_fields: dict[str, Any]
    ) -> tuple[str, str, str, dict[str, Any]]:
        """The loop control stopped the call: a duplicate, or the iteration ceiling."""
        duplicate = outcome.stage == DUPLICATE
        self._reporter_blocked(
            "loop_control",
            summary="Duplicate action suppressed"
            if duplicate
            else "Loop control aborted the action",
            event_kind="loop_control",
            fields=control_fields,
        )
        result_fields = {
            "tool": call.action,
            "result": "suppressed_duplicate" if duplicate else "loop_control_aborted",
            "execution_status": "not_executed",
        }
        self._reporter_blocked(
            "tool_result",
            summary=f"Tool not executed: {call.action}",
            event_kind="tool_result",
            fields=result_fields,
        )
        return (
            self._loop_control.reject_message(call)
            if duplicate
            else "The loop control stopped this action before execution.",
            f"{call.action}: suppressed (duplicate)"
            if duplicate
            else f"{call.action}: loop control aborted",
            "tool_suppressed",
            {
                "loop_control": control_fields,
                "loop_control_status": "blocked",
                "tool_result": result_fields,
                "tool_status": TOOL_BLOCKED,
            },
        )

    async def _refused(
        self, outcome: ToolOutcome, control_fields: dict[str, Any]
    ) -> tuple[str, str, str, dict[str, Any]]:
        """The Gatekeeper denied or escalated the call (the path already audited it)."""
        call, decision, authority = outcome.call, outcome.decision, outcome.authority
        if call is None or decision is None or authority is None:
            raise RuntimeError("A refused tool outcome carries its call, decision and authority")
        if outcome.stage == NOT_OFFERED:
            self._logger.warning("Model called a tool it was not offered", tool=call.action)
            evidence = f"'{call.action}' is not available for this request."
            result_fields = {
                "tool": call.action,
                "result": "not_offered",
                "execution_status": "not_executed",
            }
            tool_status = TOOL_BLOCKED
        elif decision.verdict == _ESCALATE:
            evidence, result_fields, tool_status = await self._refer_for_review(
                call, decision, authority
            )
        else:
            # Say that it is final. The previous wording read like a
            # transient failure, and the model answered it the way it
            # answers those — by trying again, three times, until the
            # iteration cap. A deterministic verdict does not move.
            evidence = f"That action was not permitted: {decision.reason}"
            result_fields = {
                "tool": call.action,
                "result": "not_executed",
                "execution_status": "not_executed",
            }
            tool_status = TOOL_BLOCKED
        # `_FINAL` steers the model; it is not a fact, so the evidence
        # the reply is grounded on stops before it.
        text = f"{evidence} {_FINAL}"
        if tool_status != TOOL_REFERRED:
            self._reporter_blocked(
                "tool_result",
                summary=f"Tool not executed: {call.action}",
                event_kind="tool_result",
                fields=result_fields,
            )
        return (
            text,
            f"{call.action}: {decision.verdict} ({decision.question}) — {decision.reason}",
            "tool_refused",
            {
                "loop_control": control_fields,
                "loop_control_status": "completed",
                "gatekeeper": gatekeeper_fields(decision, call),
                "gatekeeper_status": "escalated" if decision.verdict == _ESCALATE else "blocked",
                "tool_result": result_fields,
                "tool_status": tool_status,
                "evidence": evidence,
            },
        )

    async def _refer_for_review(
        self, call: Any, decision: AuthorityDecision, authority: AuthorityContext
    ) -> tuple[str, dict[str, Any], str]:
        """Turn an escalation into a queued review, and say so without the reason.

        The Gatekeeper's reason names the limit; the model — and through it
        the customer — is told only that a person will decide. A limit an
        attacker can read is a limit they can structure around.

        The queueing write is the harness's, and it runs through the
        Executor like every other side effect: the trace shows EXECUTOR and
        TOOL RESULT here as it does for any call, and the write gets the
        same idempotency key and deadline. The escalated action itself is
        never approved — `ApprovedToolCall.referral` approves only the write.

        A queued review is an outcome, not a refusal: the call did not run,
        but the referral did, and it changed the record. It is reported and
        grounded as *referred*, so "your request is with our team" is not
        read against a premise that says nothing happened.
        """
        referral = self._referral.referral_call(call) if self._referral is not None else None
        referred: ToolResult | None = None
        outcome: ExecutionOutcome | None = None
        if referral is not None:
            outcome = await self._executor.execute(
                ApprovedToolCall.referral(call, referral, decision, authority)
            )
            referred = outcome.result
        if referred is None or not referred.success:
            # The Gatekeeper's reason names the limit; the model reads only that a person decides.
            wording = get_agent_settings().review_output
            text = (
                wording.no_referral
                if referred is None
                else wording.not_queued.format(error=referred.error)
            )
            return (
                text,
                {
                    "tool": call.action,
                    "result": "human_review_required",
                    "execution_status": "not_executed",
                },
                TOOL_BLOCKED,
            )
        fields = {
            **tool_result_fields(call, referred, outcome=outcome, redact=self._security.redact),
            "referral_action": referral.action if referral is not None else "",
            "result": "human_review_required",
            "execution_status": TOOL_REFERRED,
        }
        self._done(
            "tool_result",
            event_kind="tool_result",
            summary=f"Referred for human review: {call.action}",
            fields=fields,
        )
        return f"{referred.output}.", fields, TOOL_REFERRED

    async def _after_execution(
        self,
        call: Any,
        outcome: Any,
        control_fields: dict[str, Any],
        decision_fields: dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Report one completed tool call, governed or not.

        A side-effecting call that reported success is read back before it
        is believed. "The executor returned success" and "the money moved"
        are different claims, and only the second one may ground what the
        customer is told.
        """
        result = outcome.result
        result_fields = tool_result_fields(
            call, result, outcome=outcome, redact=self._security.redact
        )
        if result.success:
            self._done(
                "tool_result",
                event_kind="tool_result",
                summary=f"Tool completed: {call.action}",
                fields=result_fields,
            )
            tool_status = TOOL_COMPLETED
            verification = await self._verify(call, result)
            if verification is not None:
                result_fields = {**result_fields, "verification": verification.summary}
                if not verification.verified:
                    # Not "failed": the call did run, and may well have
                    # done something. It is *unverified*, which means no
                    # customer-facing claim may rest on it.
                    tool_status = TOOL_UNVERIFIED
        else:
            self._done_failed(
                "tool_result",
                summary=f"Tool failed: {call.action}",
                event_kind="tool_result",
                fields=result_fields,
            )
            tool_status = TOOL_FAILED
        return (
            result.output if result.success else f"Tool failed: {result.error}",
            f"{call.action}: {'ok' if result.success else 'failed'}",
            "tool_execution" if result.success else "tool_refused",
            {
                "loop_control": control_fields,
                "loop_control_status": "completed",
                "gatekeeper": decision_fields,
                "gatekeeper_status": "completed" if decision_fields else "absent",
                "executor": result_fields,
                "tool_result": result_fields,
                "tool_status": tool_status,
            },
        )

    def _mandates_for(self, state: AgentState) -> MandateSet | None:
        """This turn's mandates, or None when no issuer is wired.

        `None` and an empty set mean different things to the Gatekeeper:
        no issuer means the question was not asked, while an empty set
        means it was asked and the answer was "nothing".
        """
        if self._mandate_issuer is None or state.customer_id is None:
            return None
        return MandateSet(
            mandates=self._mandate_issuer.mandates_for(
                str(state.customer_id),
                state.query,
                state.tool_rounds,
                # What this turn established before the model was
                # consulted. The issuer decides what of it is evidence;
                # nothing the model has since proposed is in here.
                state.metadata,
            )
        )

    async def _verify(self, call: Any, result: Any) -> VerificationReport | None:
        """Read the effect back, and record it if it did not take.

        `None` means there was nothing to verify — a read, or a call whose
        effect this agent does not describe. That is not a passed check,
        and the caller does not treat it as one.
        """
        if self._verifier is None:
            return None
        report = await self._verifier.verify(call, result)
        if report is None or report.verified:
            return report
        self._done_failed(
            "effect_verification",
            summary=f"Effect not verified: {call.action}",
            event_kind="executor",
            fields={
                "action": report.effect.action,
                "target": report.effect.target,
                "expected": report.effect.expected,
                "observed": report.observed,
                "mismatches": report.mismatches,
                "reason_code": report.reason_code,
            },
        )
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or "",
                actor=self._actor,
                action=report.effect.action,
                authority_context={"target": report.effect.target, "verified": False},
                model_version="",
                # An effect that did not take is a denied claim. The reason
                # code is the rationale's prefix: `decision` only takes
                # allow/deny/escalate, and passing the code there raised.
                decision="deny",
                rationale=f"{report.reason_code}: {report.summary}",
            )
        )
        return report
