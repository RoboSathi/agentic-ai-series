"""Executing what the model proposed — the other half of the Tool-Use Loop.

Every proposal passes the control loop, then the Gatekeeper, then the
executor; and only a result that actually succeeded becomes an
authoritative fact the reply may be grounded on.

`_execute_ungoverned` is the path taken when the Gatekeeper has been
withheld. It exists to be *seen*: the same refund that was escalated a
moment ago simply happens. The audit entry is still written, because an
ungoverned execution must leave a record rather than leaving nothing, and
no budget is committed, because no budget was consulted.
"""

from typing import Any

from agentic_commons.audit import AuditEntry
from agentic_commons.executor import (
    ApprovedToolCall,
    ExecutionOutcome,
)
from agentic_commons.gatekeeper import AuthorityContext
from agentic_commons.llm.types import Message
from agentic_commons.observability.types import utc_now
from agentic_commons.tools.types import ToolResult
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import (
    control_loop_fields,
    gatekeeper_fields,
    tool_request_fields,
    tool_result_fields,
)


class ToolUseNodes(NodeSupport):
    async def act(self, state: AgentState) -> dict[str, Any]:
        """Execute the requested tool calls, each through the Gatekeeper."""
        last = state.messages[-1] if state.messages else Message()
        requests = last.tool_calls or []
        self._report(
            "tool_execution",
            summary=f"Executing {len(requests)} requested tool(s)",
            event_kind="stage",
            fields={"requested": len(requests)},
        )
        outputs: list[Message] = []
        results: list[str] = []
        decisions: list[str] = []
        invocation_fields_history: list[dict[str, Any]] = []
        offered = set(state.metadata.get("offered_actions", []))
        for requested in requests:
            self._done(
                "tool_request",
                event_kind="tool_request",
                summary=f"Requested: {requested.name}",
                fields=tool_request_fields(requested.name, requested.arguments),
            )
            text, decision, stage, invocation_fields = await self._invoke(
                requested.name, requested.arguments, state, offered
            )
            invocation_fields_history.append(invocation_fields)
            results.append(text)
            decisions.append(decision)
            self._note(state, f"result_of_{requested.name}", text, category="tool_result")
            outputs.append(
                Message(role="tool", content=text, tool_call_id=requested.id or requested.name)
            )

            # Once a proposal escalates, autonomous execution stops for the
            # remainder of this assistant turn. Do not execute later batched
            # side effects from the same model message.
            gatekeeper = invocation_fields.get("gatekeeper")
            if isinstance(gatekeeper, dict) and gatekeeper.get("decision") == "escalate":
                remaining = len(requests) - len(invocation_fields_history)
                if remaining:
                    self._reporter_blocked(
                        "gatekeeper_evaluation",
                        summary=f"Skipped {remaining} tool proposal(s) after escalation",
                        event_kind="gatekeeper",
                        fields={
                            "decision": "escalate",
                            "reason_code": "AUTONOMY_STOPPED_AFTER_ESCALATION",
                            "skipped_requests": remaining,
                        },
                    )
                break

        self._done(
            "tool_execution",
            event_kind="tool_result",
            summary=f"Processed {len(requests)} requested tool(s)",
            fields={
                "requested": len(requests),
                "completed": sum(1 for item in invocation_fields_history if item.get("tool_status") == "completed"),
                "blocked": sum(1 for item in invocation_fields_history if item.get("tool_status") == "blocked"),
                "failed": sum(1 for item in invocation_fields_history if item.get("tool_status") == "failed"),
            },
        )
        turn_escalated = any(
            item.get("gatekeeper", {}).get("decision") == "escalate"
            for item in invocation_fields_history
            if isinstance(item.get("gatekeeper"), dict)
        )
        authoritative_results = [
            text
            for text, invocation in zip(results, invocation_fields_history, strict=True)
            if invocation.get("tool_status") == "completed"
        ]
        return {
            "messages": outputs,
            "tool_results": results,
            "authoritative_tool_results": [
                *state.authoritative_tool_results,
                *authoritative_results,
            ],
            "decisions": decisions,
            "escalated": state.escalated or turn_escalated,
            "metadata": {
                **state.metadata,
                "last_tool_invocations": invocation_fields_history,
            },
        }

    async def _execute_ungoverned(
        self,
        call: Any,
        authority: AuthorityContext,
        control_fields: dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Run a tool with nothing standing between the model and execution.

        Reachable only where the `gatekeeper` feature has been withheld,
        which is what an episode does to show the difference. Two things
        still hold: the audit entry is written, so the trail shows that an
        unauthorized execution happened rather than showing nothing at
        all; and no budget is committed, because no budget was consulted.
        """
        # No report. A stage that was not built has nothing to say, and a
        # GATEKEEPER line reading "no gatekeeper" is the screen describing
        # a control that is not there — the absence is the demonstration,
        # so the trace should show a gap where the check used to be. The
        # audit entry below still records the ungoverned execution.
        approved = ApprovedToolCall.ungoverned(call, authority)
        self._audit.record(
            AuditEntry(
                trace_id=str(authority.request_context.get("metadata", {}).get("trace_id", "")),
                actor=authority.actor,
                action=call.action,
                authority_context=authority.model_dump(mode="json"),
                decision="allow",
                rationale=approved.decision.reason,
            )
        )
        # Straight to the agent's domain adapter, with no execution
        # boundary around it: no idempotency key, so asking twice really
        # does refund twice, and no timeout, so a slow tool holds the turn
        # open. Those are the Executor's guarantees, and this is what the
        # absence of them looks like.
        outcome = await self._execute_directly(call)
        return self._after_execution(call, outcome, control_fields, decision_fields={})

    async def _execute_directly(self, call: Any) -> ExecutionOutcome:
        """Run one tool with nothing wrapped around it.

        Shaped as an `ExecutionOutcome` so the reporting path downstream
        cannot tell the difference — what changed is that no idempotency
        key was taken and no deadline applied, not how the result reads.
        `replayed` is False by construction here: replay is exactly the
        thing an idempotency guard provides, and there is none.
        """
        started = utc_now()
        if self._handler is None:
            result = ToolResult(success=False, error="No tool handler is wired for direct execution")
        else:
            result = await self._handler.execute(call)
        return ExecutionOutcome(
            result=result,
            status="completed" if result.success else "failed",
            replayed=False,
            started_at=started,
            ended_at=utc_now(),
        )

    async def _invoke(
        self, name: str, arguments: str, state: AgentState, offered: set[str]
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Validate, control, authorize, approve and execute one tool call."""
        call = self._registry.build_call(name, arguments)
        if call is None:
            blocked_fields = {
                "tool": name,
                "result": "invalid_request",
                "reason_code": "TOOL_CALL_INVALID",
                "reason": "tool is unknown or arguments failed validation",
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
                {"tool_result": blocked_fields, "tool_status": "blocked"},
            )

        call = call.model_copy(update={"params": self._security.minimize(call.params)})
        authority = AuthorityContext(
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
        )

        control_report = self._control_loop.before_action(
            call, round_number=state.tool_rounds, max_rounds=self._max_rounds
        )
        control_fields = control_loop_fields(control_report)
        if control_report.decision == "abort":
            self._reporter_blocked(
                "control_loop",
                summary=(
                    "Duplicate action suppressed"
                    if control_report.reason_code == "DUPLICATE_ACTION"
                    else "Control loop aborted the action"
                ),
                event_kind="control_loop",
                fields=control_fields,
            )
            result_fields = {
                "tool": call.action,
                "result": (
                    "suppressed_duplicate"
                    if control_report.reason_code == "DUPLICATE_ACTION"
                    else "control_loop_aborted"
                ),
                "execution_status": "not_executed",
            }
            self._reporter_blocked(
                "tool_result",
                summary=f"Tool not executed: {call.action}",
                event_kind="tool_result",
                fields=result_fields,
            )
            return (
                self._control_loop.reject_message(call)
                if control_report.reason_code == "DUPLICATE_ACTION"
                else "The control loop stopped this action before execution.",
                (
                    f"{call.action}: suppressed (duplicate)"
                    if control_report.reason_code == "DUPLICATE_ACTION"
                    else f"{call.action}: control loop aborted"
                ),
                "tool_suppressed",
                {
                    "control_loop": control_fields,
                    "control_loop_status": "blocked",
                    "tool_result": result_fields,
                    "tool_status": "blocked",
                },
            )

        if self._gatekeeper is None:
            return await self._execute_ungoverned(call, authority, control_fields)

        self._report(
            "gatekeeper_evaluation",
            event_kind="gatekeeper",
            summary="Checking deterministic execution admissibility",
            fields={"requested_action": call.action},
        )
        decision = await self._gatekeeper.evaluate(call, authority)
        decision_fields = gatekeeper_fields(decision, call)
        if not decision.allowed:
            self._reporter_blocked(
                "gatekeeper_evaluation",
                summary=f"Gatekeeper: {decision.verdict.upper()}",
                event_kind="gatekeeper",
                fields=decision_fields,
            )
            if decision.question == "offered":
                self._logger.warning("Model called a tool it was not offered", tool=call.action)
                text = f"'{call.action}' is not available for this request."
                result_fields = {
                    "tool": call.action,
                    "result": "not_offered",
                    "execution_status": "not_executed",
                }
            elif decision.verdict == "escalate":
                text = f"That action requires human review: {decision.reason}"
                result_fields = {
                    "tool": call.action,
                    "result": "human_review_required",
                    "execution_status": "not_executed",
                }
            else:
                text = f"That action was not permitted: {decision.reason}"
                result_fields = {
                    "tool": call.action,
                    "result": "not_executed",
                    "execution_status": "not_executed",
                }
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
                    "control_loop": control_fields,
                    "control_loop_status": "completed",
                    "gatekeeper": decision_fields,
                    "gatekeeper_status": "blocked",
                    "tool_result": result_fields,
                    "tool_status": "blocked",
                },
            )

        self._done(
            "gatekeeper_evaluation",
            event_kind="gatekeeper",
            summary="Gatekeeper: ALLOW",
            fields=decision_fields,
        )
        approved = ApprovedToolCall.from_decision(call, decision, authority)
        outcome = await self._executor.execute(approved)
        if outcome.result.success and not outcome.replayed:
            # Budget is charged only for a real completion, not a replay.
            self._gatekeeper.commit(call, authority)
        return self._after_execution(call, outcome, control_fields, decision_fields)

    def _after_execution(
        self,
        call: Any,
        outcome: Any,
        control_fields: dict[str, Any],
        decision_fields: dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Report one completed tool call, governed or not."""
        result = outcome.result
        result_fields = tool_result_fields(call, result, outcome=outcome)
        if result.success:
            self._control_loop.record_success(call, result.output)
            self._done(
                "tool_result",
                event_kind="tool_result",
                summary=f"Tool completed: {call.action}",
                fields=result_fields,
            )
            tool_status = "completed"
        else:
            self._done_failed(
                "tool_result",
                summary=f"Tool failed: {call.action}",
                event_kind="tool_result",
                fields=result_fields,
            )
            tool_status = "failed"
        return (
            result.output if result.success else f"Tool failed: {result.error}",
            f"{call.action}: {'ok' if result.success else 'failed'}",
            "tool_execution" if result.success else "tool_refused",
            {
                "control_loop": control_fields,
                "control_loop_status": "completed",
                "gatekeeper": decision_fields,
                "gatekeeper_status": "completed" if decision_fields else "absent",
                "executor": result_fields,
                "tool_result": result_fields,
                "tool_status": tool_status,
            },
        )
