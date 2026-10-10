"""The ops agent's nodes: propose ⇄ execute, then ground.

The loop is typed tool use, not ReAct: the model proposes validated tool calls,
there is no thought step. Every card is the platform's (`StepEvents`), so a step
here reads exactly as it does on the support agent.

propose  the model reads the conversation and either asks for tools or answers
         (LLM REQUEST, LLM RESPONSE, then LOOP CONTROL)
execute  each tool call runs the governed way (`GovernedCalls`): TOOL REQUEST,
         GATEKEEPER, EXECUTOR, MCP CALL, TOOL RESULT; results go back
ground   the hallucination check — level 1, token confidence (the platform's);
         level 2, every figure found in a query result (this agent's) — then the
         reply: the answer, any unverified figures, the SQL and its rows
"""

import time
from typing import Any

from agent_platform.gatekeeper.types import AuthorityDecision
from agent_platform.hallucination_protection.confidence import ConfidenceCheck, report_confidence
from agent_platform.hallucination_protection.levels import CheckLevel, grounding_level
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message, ModelToolCall
from agent_platform.logging import get_logger, instrument
from agent_platform.loop_control import LoopControlReport
from agent_platform.observability import StageReporter, checks
from agent_platform.observability.step_events import StepEvents
from agent_platform.observability.tool_fields import display_tool_name, tool_result_fields
from agent_platform.observability.types import StatusKind
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

PROPOSE, EXECUTE, GROUND = "propose", "execute", "ground"
# Why the loop chose its next step (wording: `observability.step_events.loop_control`).
_TOOL_REQUESTED, _NO_TOOL_CALL, _ITERATION_LIMIT = (
    "tool_requested",
    "no_tool_call",
    "iteration_limit",
)
_ABORT = "abort"
# A figure's verdict, in the claim words every agent's HALLUCINATION CHECK card reads.
_SUPPORTED, _UNVERIFIED = "supported", "undetermined"
# A prefetch read takes no arguments.
_NO_ARGUMENTS = "{}"


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
        self._steps = StepEvents(reporter, max_steps=settings.max_steps, model=llm.model)
        self._figures = figures
        # Level 1 of the hallucination check (token confidence), before the figures (level 2).
        self._confidence = confidence
        self._logger = get_logger("ops_agent")
        self._guard_log = get_logger("guardrails")

    async def propose(self, state: OpsState) -> dict[str, Any]:
        """One step: the model reads the conversation and asks for tools, or answers."""
        step = state["steps"] + 1
        self._steps.begin(step)
        request = CompletionRequest(
            messages=state["messages"],
            tools=self._registry.to_openai_tools(),
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )
        purpose = self._settings.cards.purpose
        self._steps.llm_request(request, purpose=purpose)
        result = await self._llm.complete(request)
        self._steps.llm_response(result, purpose=purpose)
        # The platform loop guard owns the step cap, so the turn's loop abort is
        # reported the same way as every other agent's; routing reads the same rule.
        wants_tools = bool(result.tool_calls)
        self._report_loop(step, wants_tools, self._calls.check_step(step, wants_tools))
        reply = Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
        return {"messages": [reply], "steps": step, "answer": result.content}

    def after_propose(self, state: OpsState) -> str:
        """Tools while the model asks for them and steps remain; then the answer."""
        last = state["messages"][-1]
        return EXECUTE if last.tool_calls and state["steps"] < self._settings.max_steps else GROUND

    def _report_loop(self, step: int, wants_tools: bool, report: LoopControlReport) -> None:
        """LOOP CONTROL: the step cap and whether the model asked for tools, and what follows."""
        stopped = report.decision == _ABORT
        next_step = EXECUTE if wants_tools and not stopped else GROUND
        because = _ITERATION_LIMIT if stopped else _TOOL_REQUESTED if wants_tools else _NO_TOOL_CALL
        fields = self._steps.loop_control(
            round_number=step,
            criteria={
                "model_requested_tools": wants_tools,
                "iteration_limit": f"{step} of {self._settings.max_steps}",
            },
            next_step=next_step,
            because=because,
        )
        self._logger.info("Loop control status", **fields)

    def begin_turn(self) -> None:
        """A new question: the loop forgets the last one (before any prefetch read runs)."""
        self._calls.begin_turn()

    def can_prefetch(self, name: str) -> bool:
        """Whether a procedure's read can run before the model: known, read-only, no arguments."""
        tool = self._registry.get(name)
        return tool is not None and not tool.side_effecting and not _required(tool)

    async def prefetch(
        self, planned: list[tuple[str, str]], session_id: str
    ) -> tuple[list[Message], list[str]]:
        """A procedure's first reads, run before the model is asked: the messages, and what ran.

        Each is shown and governed like a model's call, and handed to the model as one, so it
        starts where the procedure's first steps end. A read that needs arguments, or would
        change something, is skipped.
        """
        messages: list[Message] = []
        ran: list[str] = []
        for procedure, name in planned:
            tool = self._registry.get(name)
            if tool is None or not self.can_prefetch(name):
                continue
            call = ModelToolCall(
                id=self._settings.prefetch_call_id.format(tool=tool.model_name),
                name=tool.model_name,
                arguments=_NO_ARGUMENTS,
            )
            self._steps.tool_request(call.name, call.arguments, prefetched_for=procedure)
            outcome = await self._calls.run(call, session_id, 0, on_decision=self._decided)
            self._report_call(call.name, outcome)
            messages += [
                Message(role="assistant", content="", tool_calls=[call]),
                Message(role="tool", content=outcome.text, tool_call_id=call.id),
            ]
            ran += [call.name] if outcome.succeeded else []
        return messages, ran

    async def execute(self, state: OpsState) -> dict[str, Any]:
        replies: list[Message] = []
        queries: list[QueryRecord] = []
        failures = 0
        calls = state["messages"][-1].tool_calls or []
        for call in calls:
            self._steps.tool_request(call.name, call.arguments)
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
        self._steps.begin(None)
        self._steps.hallucination_check(status, outcome, fields)

    def _decided(self, call: ToolCall, decision: AuthorityDecision) -> None:
        """The Gatekeeper's ruling on one call, as it ruled (the audit entry is its own)."""
        self._steps.gatekeeper(call, decision)

    def _report_call(self, name: str, outcome: CallOutcome) -> None:
        """The call's result, named for what came back, with the tool's own preview."""
        cards = self._settings.cards
        tool = display_tool_name(name)
        path = outcome.path
        if path is None or path.call is None or path.execution is None:
            stopped = path.stage if path is not None else ""
            fields = {"tool": tool, "result": stopped, "reason": outcome.text}
            # Only the Gatekeeper refuses; a repeat or a malformed call was not run, not refused.
            if outcome.refused:
                self._steps.tool_result("blocked", cards.refused.format(tool=tool), fields)
            else:
                summary = cards.not_run.format(tool=tool, stage=stopped)
                self._steps.tool_result("warning", summary, fields)
            return
        result = path.execution.result
        fields = tool_result_fields(path.call, result, outcome=path.execution)
        fields.pop("output", None)  # the structured preview below says it more readably
        if not outcome.succeeded:
            self._steps.tool_result("failed", cards.failed.format(tool=tool), fields, result=result)
            return
        summary, preview = result_card(tool, outcome.data, self._settings)
        self._steps.tool_result("completed", summary, {**fields, **preview}, result=result)


def _claim_check(finding: FigureFinding, found: str) -> dict[str, Any]:
    """A figure as the card's claim line: supported by a query's result, or unverified."""
    if not finding.supported:
        return {"claim": finding.figure, "verdict": _UNVERIFIED}
    rationale = found.format(query=finding.query)
    return {"claim": finding.figure, "verdict": _SUPPORTED, "rationale": rationale}


__all__ = ["EXECUTE", "GROUND", "PROPOSE", "OpsNodes"]


def _required(tool: Any) -> bool:
    """Whether a tool needs arguments a prefetch has no way to know."""
    return any(field.is_required() for field in tool.args_schema.model_fields.values())
