"""The direct answer as a ReAct loop: THOUGHT (one model call) -> ACTION (the tools it asked for)
-> OBSERVATION (what they returned) -> THOUGHT ... then the HALLUCINATION CHECK and the ANSWER.

    START -> reason -> act -> reason ... -> ground -> finalize -> END

`reason` ends the loop by answering; `act` hands the results back. A budget decides every other
exit: when the last step, the tool-call cap, the token cap or the clock is reached the model is
asked once more *without tools*, so it answers from what it has instead of being cut off mid-search.
`ground` is the HALLUCINATION CHECK: the answer is checked against what the tools returned, and only if that
check refutes a claim is the model asked once to REVISE it. Every step is a card on the live feed,
numbered "Step N", and every payload is in the log and the audit trail.
"""

import inspect
import json
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent_platform.context.types import Chunk
from agent_platform.hallucination_protection import Observation, tool_evidence
from agent_platform.llm.embeddings import tokenize
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message, ModelToolCall
from agent_platform.logging import describe_error, get_logger, instrument
from agent_platform.observability.llm_fields import llm_request_fields, llm_response_fields
from agent_platform.observability.tool_fields import display_tool_name, tool_request_fields
from agent_platform.tools.registry import NamespacedToolRegistry
from agent_platform.tools.types import ToolResult
from research_agent.config import get_agent_settings
from research_agent.config.settings import AnswerSettings
from research_agent.domain.budget import TOKENS
from research_agent.graph.answer_state import STEPS, TOOL_CALLS, AnswerState, Turn, turn_of
from research_agent.graph.points import as_points
from research_agent.guardrails.answer_grounding import AnswerGrounding, GroundingOutcome
from research_agent.llm.structured import ModelCallTimeoutError, StructuredCaller
from research_agent.presentation.narration import clip
from research_agent.prompts import answer as prompt
from research_agent.research.telemetry import RunTelemetry
from research_agent.tools import URLS_KEY

# The loop's cards, in the order a step emits them.
THOUGHT, ACTION, OBSERVATION, REVISE, ANSWER = (
    "thought",
    "action",
    "observation",
    "revise",
    "answer",
)
HALLUCINATION_CHECK = "hallucination_check"
LOOP_STAGES = frozenset({THOUGHT, ACTION, OBSERVATION})
# Why a model call was made, as the card and the log say it.
ANSWER_PURPOSE, REVISE_PURPOSE = "answer", "grounding_repair"
_URL = re.compile(r"https?://\S+")


@instrument("research_agent")
class AnswerNodes:
    """The loop's nodes and edges."""

    def __init__(
        self,
        caller: StructuredCaller,
        registry: NamespacedToolRegistry,
        settings: AnswerSettings,
        telemetry: RunTelemetry,
        grounding: AnswerGrounding | None = None,
    ) -> None:
        self._grounding = grounding
        self._caller = caller
        self._registry = registry
        self._settings = settings
        self._notes = settings.messages
        self._telemetry = telemetry
        self._sources = re.compile(settings.grounding.sources_pattern)
        self._logger = get_logger("research_agent")

    # -- THOUGHT ----------------------------------------------------------------------------

    async def reason(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        turn = turn_of(config)
        turn.budget.charge(STEPS, 1)
        turn.loop.begin(int(turn.spent(STEPS)))
        limit = self._limit(turn)
        if limit == "seconds":
            # Too little time left for a model call: answer with what there is.
            return {"stop": limit}
        tools = None if limit else self._registry.to_openai_tools()
        try:
            result = await self._think(turn, state["messages"], tools)
        except Exception as exc:
            return self._failed(turn, exc)
        wanted = (
            (result.tool_calls or [])[: int(turn.budget.remaining(TOOL_CALLS))] if tools else []
        )
        if wanted:
            return {
                "messages": [Message(role="assistant", content=result.content, tool_calls=wanted)]
            }
        used = turn.spent(TOOL_CALLS) > 0
        return {"text": result.content.strip(), "stop": limit if limit and used else ""}

    async def _think(
        self, turn: Turn, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> CompletionResult:
        """One reasoning step: the model call, shown as one THOUGHT card."""
        request = CompletionRequest(messages=messages, tools=tools)
        self._emit(turn, THOUGHT, self._waiting(), {}, event_kind="model", status="started")
        result = await self._call(turn, messages, tools, self._call_timeout(turn))
        calls = ", ".join(
            self._notes.tool_call.format(tool=display_tool_name(c.name), arguments=c.arguments)
            for c in result.tool_calls or []
        )
        heading = (self._notes.thought_tools if calls else self._notes.thought_answer).format(
            calls=calls, **self._usage(result)
        )
        fields = {
            **self._exchange_fields(request, result, ANSWER_PURPOSE),
            **(self._context_fields(turn) if turn.spent(STEPS) == 1 else {}),
        }
        self._emit(turn, THOUGHT, heading, fields, event_kind="model", tokens=result)
        return result

    async def _call(
        self,
        turn: Turn,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        timeout: float,
    ) -> CompletionResult:
        """The model call itself, charged to the turn. Its usage goes on the step's card."""
        result = await self._caller.chat(
            self._settings.role,
            messages,
            tools,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            timeout=timeout,
            quiet=True,
        )
        turn.budget.charge(TOKENS, result.usage.total_tokens)
        return result

    def _exchange_fields(
        self, request: CompletionRequest, result: CompletionResult, purpose: str
    ) -> dict[str, Any]:
        """What the model was handed and what it gave back, as the support agent's cards show it."""
        model = self._caller.model_key(self._settings.role)
        return {
            **llm_request_fields(request, model=model),
            **llm_response_fields(result),
            "purpose": purpose,
            **({"reasoning": result.reasoning} if result.reasoning else {}),
        }

    def _context_fields(self, turn: Turn) -> dict[str, Any]:
        """What the session added to the question, on the first step only, and only if anything."""
        context, notes = turn.context, self._notes
        lines = [
            *([notes.context_turns.format(turns=context.turns)] if context.turns else []),
            *([notes.context_runs.format(runs=", ".join(context.runs))] if context.runs else []),
        ]
        return {"session_context": lines} if lines else {}

    @staticmethod
    def _usage(result: CompletionResult) -> dict[str, Any]:
        return {
            "tokens": f"{result.usage.total_tokens:,}",
            "seconds": result.latency_ms / 1000,
        }

    def _limit(self, turn: Turn) -> str:
        """The cap that forbids another tool call now ("" while tools are still allowed)."""
        budget = turn.budget
        if budget.expired() or self._time_left(turn) < self._settings.min_call_seconds:
            return "seconds"
        if budget.remaining(TOKENS) <= 0:
            return TOKENS
        if budget.remaining(STEPS) <= 1:
            return STEPS
        if budget.remaining(TOOL_CALLS) <= 0:
            return TOOL_CALLS
        return ""

    def _time_left(self, turn: Turn) -> float:
        return self._settings.budget.max_seconds - turn.budget.elapsed()

    def _call_timeout(self, turn: Turn) -> float:
        return min(self._settings.call_timeout_seconds, self._time_left(turn))

    def _waiting(self) -> str:
        """The pending THOUGHT card's heading, shown while the model call is in flight."""
        return self._notes.thought_waiting.format(role=self._settings.role)

    def _failed(self, turn: Turn, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ModelCallTimeoutError):
            cause = self._notes.model_timeout.format(seconds=exc.seconds)
        else:
            cause = describe_error(exc)
        self._logger.error("Direct answer step failed", cause=cause, question=turn.question)
        # Settles the pending THOUGHT card, which would otherwise wait forever.
        self._emit(turn, THOUGHT, cause, {}, event_kind="model", status="failed")
        self._telemetry.audit("answer.failed", "deny", cause, question=turn.question)
        return {"failure": cause}

    def after_reason(self, state: AnswerState) -> str:
        if state["failure"]:
            return "finalize"
        last = state["messages"][-1]
        return "act" if last.role == "assistant" and last.tool_calls else "ground"

    # -- ACTION and OBSERVATION -------------------------------------------------------------

    async def act(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        turn = turn_of(config)
        replies: list[Message] = []
        sources: list[str] = []
        observations: list[Observation] = []
        for call in state["messages"][-1].tool_calls or []:
            reply, urls, seen = await self._run(call, turn)
            replies.append(reply)
            sources.extend(urls)
            observations.extend(seen)
        return {"messages": replies, "sources": sources, "observations": observations}

    async def _run(
        self, call: ModelToolCall, turn: Turn
    ) -> tuple[Message, list[str], list[Observation]]:
        turn.budget.charge(TOOL_CALLS, 1)
        name = display_tool_name(call.name)
        self._emit(
            turn,
            ACTION,
            self._notes.tool_call.format(tool=name, arguments=call.arguments),
            tool_request_fields(call.name, call.arguments),
            event_kind="tool_request",
        )
        result = await self._execute(call, turn)
        text = result.output if result.success else result.error or self._notes.empty
        self._observe(turn, name, text, result)
        self._telemetry.audit(
            "answer.tool",
            "allow" if result.success else "deny",
            f"{call.name}: {'ok' if result.success else 'failed'}",
            arguments=call.arguments,
            result=text,
        )
        urls = [str(url) for url in result.data.get(URLS_KEY, [])]
        # Only what a tool actually returned can ground a claim; an error or a refusal cannot.
        seen = [Observation(text, source=urls[0] if urls else call.name)] if result.success else []
        return Message(role="tool", content=text, tool_call_id=call.id), urls, seen

    async def _execute(self, call: ModelToolCall, turn: Turn) -> ToolResult:
        """The tool, after the watchdog: an unknown tool or bad arguments are said, not raised."""
        tool = self._registry.get(call.name)
        built = self._registry.build_call(call.name, call.arguments)
        if tool is None or built is None:
            text = self._notes.unknown_tool if tool is None else self._notes.bad_arguments
            return ToolResult(error=text.format(tool=call.name))
        report = turn.control.before_action(
            built,
            round_number=int(turn.spent(STEPS)),
            max_rounds=self._settings.budget.max_steps,
        )
        if report.decision == "abort":
            return ToolResult(success=True, output=turn.control.reject_message(built))
        result = await self._invoke(tool.handler, built.params, call.name)
        if result.success:
            turn.control.record_success(built, result.output)
        else:
            turn.control.record_refusal(built, result.error or "")
        return result

    def _observe(self, turn: Turn, tool: str, text: str, result: ToolResult) -> None:
        limit = get_agent_settings().tui.clip_chars
        lines = [clip(line, limit) for line in text.splitlines() if line.strip()]
        heading = (
            self._notes.observation_heading.format(tool=tool, lines=len(lines))
            if result.success
            else self._notes.observation_failed.format(tool=tool, error=clip(text, limit))
        )
        fields = {
            "tool": tool,
            "result": "success" if result.success else "failed",
            "observation": lines or [self._notes.no_results],
        }
        status = "completed" if result.success else "failed"
        self._emit(turn, OBSERVATION, heading, fields, event_kind="tool_result", status=status)

    async def _invoke(self, handler: Any, params: Mapping[str, Any], name: str) -> ToolResult:
        try:
            outcome = handler(**params)
            return await outcome if inspect.isawaitable(outcome) else outcome  # type: ignore[no-any-return]
        except Exception as exc:
            detail = describe_error(exc)
            self._logger.warning("Answer tool failed", tool=name, error=detail)
            return ToolResult(error=self._notes.tool_failed.format(tool=name, error=detail))

    def after_act(self, state: AnswerState, config: RunnableConfig) -> str:
        """Back to the model while any budget remains; an exhausted one ends the loop."""
        budget = turn_of(config).budget
        spent = budget.exhausted_counters()
        return "finalize" if budget.expired() or STEPS in spent or TOKENS in spent else "reason"

    # -- HALLUCINATION CHECK and REVISE -----------------------------------------------------------------

    async def ground(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        """Check an answer that rests on tool results against those results.

        An answer given without a tool call rests on the model's own knowledge, which nothing
        here can check; the prompt asks it to use a tool for anything it is unsure of.
        """
        if self._grounding is None or not state["observations"] or not state["text"]:
            return {}
        turn = turn_of(config)
        evidence = tool_evidence(state["observations"])
        outcome = await self._grounding.check(
            turn.question, state["text"], evidence, self._reviser(turn, evidence)
        )
        self._card_check(turn, outcome)
        return {"text": outcome.text}

    def _reviser(self, turn: Turn, evidence: list[Chunk]) -> Any:
        """The model's one rewrite of the claims the check refuted, from the evidence about them.

        It is handed the answer, the refuted claims and, for each, the few evidence lines that
        bear on it — not the session, not every page the tools read.
        """

        async def revise(candidate: str, unsupported: list[str]) -> str:
            if self._time_left(turn) < self._settings.min_call_seconds:
                return candidate  # no time: the guardrail removes what it refuted instead
            payload = {
                "answer": candidate,
                "unsupported": unsupported,
                "evidence": self._evidence_for(unsupported, evidence),
            }
            messages = [
                Message(role="system", content=prompt.REVISE_INSTRUCTIONS),
                Message(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ]
            timeout = min(self._settings.repair_timeout_seconds, self._time_left(turn))
            result = await self._call(turn, messages, None, timeout)
            heading = self._notes.revise_heading.format(
                claims=len(unsupported), **self._usage(result)
            )
            request = CompletionRequest(messages=messages)
            fields = self._exchange_fields(request, result, REVISE_PURPOSE)
            self._emit(turn, REVISE, heading, fields, event_kind="model", tokens=result)
            return result.content.strip() or candidate

        return revise

    def _evidence_for(self, claims: list[str], evidence: list[Chunk]) -> list[str]:
        """The evidence lines that share the most words with each claim, best first."""
        per_claim = self._settings.grounding.repair_evidence_per_claim
        chosen: list[str] = []
        for claim in claims:
            words = set(tokenize(claim))
            ranked = sorted(evidence, key=lambda c: -len(words & set(tokenize(c.content))))
            chosen.extend(c.content for c in ranked[:per_claim] if c.content not in chosen)
        return chosen

    def _card_check(self, turn: Turn, outcome: GroundingOutcome) -> None:
        tally = str(outcome.report.get("tally", ""))
        heading = (
            self._notes.grounded_heading.format(tally=tally)
            if outcome.clean
            else self._notes.ungrounded_heading.format(removed=len(outcome.removed), tally=tally)
        )
        status = "completed" if outcome.clean else "warning"
        self._emit(
            turn,
            HALLUCINATION_CHECK,
            heading,
            outcome.report,
            event_kind="grounding",
            status=status,
        )

    # -- ANSWER -----------------------------------------------------------------------------

    async def finalize(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        turn = turn_of(config)
        steps, seconds = int(turn.spent(STEPS)), turn.budget.elapsed()
        if state["failure"]:
            text = self._notes.failed.format(cause=state["failure"], step=steps, seconds=seconds)
            self._card_answer(turn, text, failed=True)
            return {"text": text}
        text = state["text"] or self._notes.empty
        points = self._settings.pointwise
        if points.enabled:
            text = as_points(text, self._notes.sources_heading, points.marker, points.min_sentences)
        if state["sources"] and not _URL.search(text):
            # The model's own Sources line named no URL ("weather data for Pune"): the URLs the
            # tools actually read replace it.
            unique = list(dict.fromkeys(state["sources"]))
            body = self._sources.split(text, maxsplit=1)[0].rstrip()
            text = f"{body}\n\n{self._notes.sources_heading}\n" + "\n".join(
                f"- {url}" for url in unique
            )
        stop = state["stop"] or (turn.budget.exhausted_counters() or [""])[0]
        if stop and (not state["text"] or turn.spent(TOOL_CALLS) > 0):
            text = f"{text}\n\n{self._notes.cap_reached.format(counter=stop)}"
        self._card_answer(turn, text, failed=False)
        self._telemetry.audit(
            "answer.completed",
            "allow",
            "answered directly",
            question=turn.question,
            steps=steps,
            tool_calls=int(turn.spent(TOOL_CALLS)),
            stop=stop,
            answer=text,
        )
        return {"text": text}

    def _card_answer(self, turn: Turn, text: str, failed: bool) -> None:
        heading = (
            text.splitlines()[0]
            if failed
            else self._notes.answered.format(
                steps=int(turn.spent(STEPS)),
                calls=int(turn.spent(TOOL_CALLS)),
                tokens=f"{int(turn.spent(TOKENS)):,}",
                seconds=turn.budget.elapsed(),
            )
        )
        status = "failed" if failed else "completed"
        self._emit(turn, ANSWER, heading, {"response": text.strip()}, status=status)

    # -- cards ------------------------------------------------------------------------------

    def _emit(
        self,
        turn: Turn,
        stage: str,
        heading: str,
        fields: Mapping[str, Any],
        *,
        event_kind: Any = "stage",
        status: Any = "completed",
        tokens: CompletionResult | None = None,
    ) -> None:
        """One card of this turn, numbered by its step when it belongs to the loop."""
        turn.loop.emit(
            stage,
            status,
            summary=heading,
            fields=fields,
            event_kind=event_kind,
            tokens=tokens.usage.total_tokens if tokens is not None else None,
        )


__all__ = [
    "ANSWER",
    "ACTION",
    "LOOP_STAGES",
    "HALLUCINATION_CHECK",
    "OBSERVATION",
    "THOUGHT",
    "AnswerNodes",
]
