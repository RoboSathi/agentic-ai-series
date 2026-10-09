"""The ops agent's nodes: reason ⇄ act, then ground.

reason   the model reads the conversation and either asks for tools or answers
         (LLM REQUEST, then LLM RESPONSE: the support agent's cards)
act      each tool call runs the governed way (`GovernedCalls`); results go back
ground   the hallucination check — level 1, token confidence (the platform's);
         level 2, every figure found in a query result (this agent's) — then the
         reply: the answer, any unverified figures, the SQL and its rows
"""

import time
from typing import Any

from agent_platform.config import get_settings
from agent_platform.gatekeeper.types import AuthorityDecision
from agent_platform.hallucination_protection.confidence import ConfidenceCheck, report_confidence
from agent_platform.hallucination_protection.levels import CheckLevel, grounding_level
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message, ModelToolCall
from agent_platform.logging import get_logger, instrument
from agent_platform.observability import StageReporter, checks
from agent_platform.observability.llm_fields import llm_request_fields, llm_response_fields
from agent_platform.observability.tool_fields import (
    display_tool_name,
    tool_request_fields,
    tool_result_fields,
)
from agent_platform.observability.types import EventKind, StatusKind
from agent_platform.observability.units import report_grounding
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.types import ToolCall
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.domain import QueryRecord
from ops_agent.graph.state import OpsState
from ops_agent.graph.tool_calls import CallOutcome, GovernedCalls
from ops_agent.guardrails import FigureCheck, FigureFinding
from ops_agent.presentation import present
from ops_agent.presentation.cards import result_card

ACT, GROUND = "act", "ground"
# A figure's verdict, in the claim words every agent's HALLUCINATION CHECK card reads.
_SUPPORTED, _UNVERIFIED = "supported", "undetermined"


@instrument("ops_agent")
class OpsNodes:
    """The graph's nodes, holding what each needs."""

    def __init__(
        self,
        llm: LLMClient,
        registry: NamespacedToolRegistry,
        calls: GovernedCalls,
        settings: OpsAgentSettings,
        reporter: StageReporter | None = None,
        figures: FigureCheck | None = None,
        confidence: ConfidenceCheck | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._calls = calls
        self._settings = settings
        self._reporter = reporter
        self._figures = figures
        # Level 1 of the hallucination check (token confidence), before the figures (level 2).
        self._confidence = confidence
        self._logger = get_logger("ops_agent")
        self._guard_log = get_logger("guardrails")

    async def reason(self, state: OpsState) -> dict[str, Any]:
        """One step: the model reads the conversation and asks for tools, or answers."""
        if state["steps"] == 0:
            self._calls.begin_turn()
        step = state["steps"] + 1
        request = CompletionRequest(
            messages=state["messages"],
            tools=self._registry.to_openai_tools(),
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )
        self._report_request(request)
        result = await self._llm.complete(request)
        self._report_response(result)
        # The platform loop guard owns the step cap, so the turn's loop abort is
        # reported the same way as every other agent's; routing reads the same rule.
        self._calls.check_step(step, bool(result.tool_calls))
        reply = Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
        return {"messages": [reply], "steps": step, "answer": result.content}

    def after_reason(self, state: OpsState) -> str:
        """Tools while the model asks for them and steps remain; then the answer."""
        last = state["messages"][-1]
        return ACT if last.tool_calls and state["steps"] < self._settings.max_steps else GROUND

    async def act(self, state: OpsState) -> dict[str, Any]:
        replies: list[Message] = []
        queries: list[QueryRecord] = []
        failures = 0
        calls = state["messages"][-1].tool_calls or []
        for call in calls:
            self._requested(call)
            outcome = await self._calls.run(
                call, state["session_id"], state["steps"], on_decision=self._decided
            )
            self._report_call(call.name, outcome)
            replies.append(Message(role="tool", content=outcome.text, tool_call_id=call.id))
            failures += 0 if outcome.succeeded else 1
            if outcome.query is not None:
                queries.append(outcome.query)
        return {
            "messages": replies,
            "queries": queries,
            "tool_calls": state["tool_calls"] + len(calls),
            "tool_failures": state["tool_failures"] + failures,
        }

    async def ground(self, state: OpsState) -> dict[str, Any]:
        last = state["messages"][-1]
        answer = (
            self._settings.words.stopped.format(steps=state["steps"])
            if last.tool_calls
            else state["answer"]
        )
        unsupported = self._check(answer, state["queries"])
        return {
            "answer": answer,
            "response": present(answer, state["queries"], unsupported, self._settings),
            "unverified": None if self._figures is None else len(unsupported),
        }

    def _check(self, answer: str, queries: list[QueryRecord]) -> list[str]:
        """The hallucination check: level 1 (token confidence), then level 2 (the figures)."""
        if self._figures is None:
            return []
        first = self._first_level(answer)
        started = time.perf_counter()
        findings = self._figures.findings(answer, queries)
        unsupported = [finding.figure for finding in findings if not finding.supported]
        # The figures are this agent's claims: the turn's groundedness is read from them.
        figures = len(findings)
        report_grounding(figures, figures - len(unsupported), len(unsupported))
        elapsed = (time.perf_counter() - started) * 1000
        checks.record_grounding(checks.RULE, "", figures, len(unsupported), elapsed)
        self._report_check(first, findings, len(queries))
        return unsupported

    def _first_level(self, answer: str) -> list[CheckLevel]:
        """Level 1, with the note that it changes no verdict here (none when withheld)."""
        if self._confidence is None or not self._confidence.enabled:
            return []
        level = report_confidence(self._confidence.assess(answer))
        note = self._settings.cards.confidence_signal
        return [level.model_copy(update={"lines": [*level.lines, note]})]

    def _report_check(
        self, first: list[CheckLevel], findings: list[FigureFinding], queries: int
    ) -> None:
        """One HALLUCINATION CHECK card and log line, the support agent's shape, levels first."""
        cards = self._settings.cards
        missing = sum(not finding.supported for finding in findings)
        tally = (
            cards.figure_tally.format(supported=len(findings) - missing, figures=len(findings))
            if findings
            else cards.no_figures
        )
        second = grounding_level(cards.figure_level, tally, flagged=bool(missing))
        outcome = cards.ungrounded.format(count=missing) if missing else cards.grounded
        fields: dict[str, Any] = {
            "levels": [level.model_dump() for level in [*first, second]],
            "outcome": outcome,
            "tally": tally,
            "claim_checks": [_claim_check(finding, cards.figure_found) for finding in findings],
            "evidence_searched": cards.queries_searched.format(count=queries),
        }
        self._guard_log.info("Response grounding checked", **fields)
        status: StatusKind = "warning" if missing else "completed"
        self._card(status, self._settings.stages.ground, outcome, "grounding", fields)

    def _report_request(self, request: CompletionRequest) -> None:
        """LLM REQUEST: what the model is handed this step (prompt, messages, tools offered)."""
        purpose = self._settings.cards.purpose
        summary = get_settings().observability.llm_request_heading.format(purpose=purpose)
        fields = {**llm_request_fields(request, model=self._llm.model), "purpose": purpose}
        self._card("completed", self._settings.stages.llm_request, summary, "model", fields)

    def _report_response(self, result: CompletionResult) -> None:
        """LLM RESPONSE: what came back — the tool calls it asked for, or its answer."""
        purpose, headings = self._settings.cards.purpose, get_settings().observability
        summary = (
            headings.llm_tool_requested_heading.format(tool=result.tool_calls[0].name)
            if result.tool_calls
            else headings.llm_response_heading.format(purpose=purpose)
        )
        fields = {**llm_response_fields(result), "purpose": purpose}
        self._card("completed", self._settings.stages.llm_response, summary, "model", fields)

    def _requested(self, call: ModelToolCall) -> None:
        cards, stage = self._settings.cards, self._settings.stages.request
        fields = tool_request_fields(call.name, call.arguments)
        summary = cards.requested.format(tool=display_tool_name(call.name))
        self._card("completed", stage, summary, "tool_request", fields)

    def _decided(self, call: ToolCall, decision: AuthorityDecision) -> None:
        """The Gatekeeper's ruling on one call, as it ruled (the audit entry is its own)."""
        cards, stage = self._settings.cards, self._settings.stages.gatekeeper
        summary = cards.gatekeeper.format(
            verdict=decision.verdict.upper(), standing=decision.standing, reason=decision.reason
        )
        fields = {
            "action": call.action,
            "decision": decision.verdict,
            "standing": decision.standing,
            "question": decision.question,
            "reason_code": decision.reason_code,
            "reason": decision.reason,
        }
        status: StatusKind = "completed" if decision.allowed else "blocked"
        self._card(status, stage, summary, "gatekeeper", fields)

    def _report_call(self, name: str, outcome: CallOutcome) -> None:
        """The call's result, named for what came back, with the tool's own preview."""
        cards, stage = self._settings.cards, self._settings.stages.act
        tool = display_tool_name(name)
        path = outcome.path
        if path is None or path.call is None or path.execution is None:
            stopped = path.stage if path is not None else ""
            fields = {"tool": tool, "result": stopped, "reason": outcome.text}
            # Only the Gatekeeper refuses; a repeat or a malformed call was not run, not refused.
            if outcome.refused:
                self._card("blocked", stage, cards.refused.format(tool=tool), "tool_result", fields)
            else:
                summary = cards.not_run.format(tool=tool, stage=stopped)
                self._card("warning", stage, summary, "tool_result", fields)
            return
        fields = tool_result_fields(path.call, path.execution.result, outcome=path.execution)
        fields.pop("output", None)  # the structured preview below says it more readably
        if not outcome.succeeded:
            self._card("failed", stage, cards.failed.format(tool=tool), "tool_result", fields)
            return
        summary, preview = result_card(tool, outcome.data, self._settings)
        self._card("completed", stage, summary, "tool_result", {**fields, **preview})

    def _card(
        self,
        status: StatusKind,
        stage: str,
        summary: str,
        kind: EventKind,
        fields: dict[str, Any] | None = None,
    ) -> None:
        if self._reporter is not None:
            self._reporter.emit(stage, status, event_kind=kind, summary=summary, fields=fields)


def _claim_check(finding: FigureFinding, found: str) -> dict[str, Any]:
    """A figure as the card's claim line: supported by a query's result, or unverified."""
    if not finding.supported:
        return {"claim": finding.figure, "verdict": _UNVERIFIED}
    rationale = found.format(query=finding.query)
    return {"claim": finding.figure, "verdict": _SUPPORTED, "rationale": rationale}


__all__ = ["ACT", "GROUND", "OpsNodes"]
