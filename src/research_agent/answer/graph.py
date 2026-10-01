"""The direct answer as a ReAct loop: reason (one model call) -> act (the tools it asked for) -> reason.

    START -> reason -> act -> reason ... -> finalize -> END

`reason` ends the loop by answering; `act` hands the results back. A budget decides every other
exit: when the last step, the tool-call cap, the token cap or the clock is reached the model is
asked once more *without tools*, so it answers from what it has instead of being cut off mid-search.
Every step is a card on the live feed, and every payload is in the log and the audit trail.
"""

import inspect
import json
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agentic_commons.context.types import Chunk
from agentic_commons.control_loop import BudgetGuard
from agentic_commons.control_loop.watchdogs import ToolControlLoop
from agentic_commons.llm.types import CompletionRequest, CompletionResult, Message, ModelToolCall
from agentic_commons.logging import get_logger, instrument
from agentic_commons.observability.llm_fields import llm_request_fields, llm_response_fields
from agentic_commons.tools.registry import NamespacedToolRegistry
from agentic_commons.tools.types import ToolResult
from research_agent.answer.context import SessionContext
from research_agent.answer.grounding import AnswerGrounding, GroundingOutcome
from research_agent.answer.tools import URLS_KEY
from research_agent.config import get_agent_settings
from research_agent.config.settings import AnswerSettings
from research_agent.domain.budget import TOKENS
from research_agent.narration import clip
from research_agent.prompts import answer as prompt
from research_agent.research.telemetry import RunTelemetry
from research_agent.structured import ModelCallTimeoutError, StructuredCaller

STEPS, TOOL_CALLS = "steps", "tool_calls"
STAGE, TOOL_STAGE = "answer", "answer_tool"
# Why a model call was made, as its request card says.
ANSWER_PURPOSE, REVISE_PURPOSE = "answer", "grounding_repair"
_URL = re.compile(r"https?://\S+")
_CONFIG_KEY = "turn"


class AnswerState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    sources: Annotated[list[str], operator.add]
    # What the tools returned, as the grounding check reads it.
    evidence: Annotated[list[Chunk], operator.add]
    text: str
    # The budget counter that cut the loop short ("" when the model finished by itself).
    stop: str
    # Why no answer could be given ("" when one was).
    failure: str


@dataclass
class Turn:
    """What belongs to one question and is not state: its budget and its watchdog."""

    question: str
    budget: BudgetGuard
    control: ToolControlLoop
    context: SessionContext

    def spent(self, counter: str) -> float:
        return self.budget.snapshot()[counter].spent


def _turn(config: RunnableConfig) -> Turn:
    turn = config["configurable"][_CONFIG_KEY]
    assert isinstance(turn, Turn)
    return turn


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
        self._telemetry = telemetry
        self._logger = get_logger("research_agent")

    # -- reason ---------------------------------------------------------------------------

    async def reason(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        turn = _turn(config)
        turn.budget.charge(STEPS, 1)
        limit = self._limit(turn)
        tools = None if limit else self._registry.to_openai_tools()
        try:
            result = await self._chat(turn, state["messages"], tools, ANSWER_PURPOSE)
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

    async def _chat(
        self,
        turn: Turn,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        purpose: str,
    ) -> CompletionResult:
        """One model call, shown as the support agent shows its own: the request, then the response.

        The request card says exactly what the model was handed (the system prompt, the session
        context, the question, the tools offered); the response card says what came back.
        """
        notes = self._settings.messages
        request = CompletionRequest(messages=messages, tools=tools)
        model = self._caller.model_key(self._settings.role)
        context = turn.context
        self._telemetry.stage(
            "llm_request",
            notes.request_heading.format(purpose=purpose),
            {
                **llm_request_fields(request, model=model),
                "purpose": purpose,
                "history_turns": context.turns,
                "session_state": sorted(context.state),
                "earlier_runs": context.runs,
            },
        )
        result = await self._caller.chat(
            self._settings.role,
            messages,
            tools,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            timeout=self._call_timeout(turn),
        )
        turn.budget.charge(TOKENS, result.usage.total_tokens)
        self._telemetry.stage(
            "llm_response_received",
            notes.response_heading.format(purpose=purpose),
            {**llm_response_fields(result), "purpose": purpose},
        )
        return result

    def _limit(self, turn: Turn) -> str:
        """The cap that forbids another tool call now ("" while tools are still allowed)."""
        budget = turn.budget
        if budget.expired():
            return "seconds"
        if budget.remaining(TOKENS) <= 0:
            return TOKENS
        if budget.remaining(STEPS) <= 1:
            return STEPS
        if budget.remaining(TOOL_CALLS) <= 0:
            return TOOL_CALLS
        return ""

    def _call_timeout(self, turn: Turn) -> float:
        left = self._settings.budget.max_seconds - turn.budget.elapsed()
        return min(self._settings.call_timeout_seconds, left)

    def _failed(self, turn: Turn, exc: Exception) -> dict[str, Any]:
        notes = self._settings.messages
        if isinstance(exc, ModelCallTimeoutError):
            cause = notes.model_timeout.format(seconds=exc.seconds)
        else:
            cause = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        self._logger.error("Direct answer step failed", cause=cause, question=turn.question)
        self._telemetry.audit("answer.failed", "deny", cause, question=turn.question)
        return {"failure": cause}

    def after_reason(self, state: AnswerState) -> str:
        if state["failure"]:
            return "finalize"
        last = state["messages"][-1]
        return "act" if last.role == "assistant" and last.tool_calls else "ground"

    # -- act ------------------------------------------------------------------------------

    async def act(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        turn = _turn(config)
        replies: list[Message] = []
        sources: list[str] = []
        evidence: list[Chunk] = []
        for call in state["messages"][-1].tool_calls or []:
            reply, urls, chunk = await self._run(call, turn)
            replies.append(reply)
            sources.extend(urls)
            if chunk is not None:
                evidence.append(chunk)
        return {"messages": replies, "sources": sources, "evidence": evidence}

    async def _run(
        self, call: ModelToolCall, turn: Turn
    ) -> tuple[Message, list[str], Chunk | None]:
        turn.budget.charge(TOOL_CALLS, 1)
        notes = self._settings.messages
        tool = self._registry.get(call.name)
        heading = notes.tool_call.format(
            tool=tool.name if tool else call.name, arguments=call.arguments
        )
        built = self._registry.build_call(call.name, call.arguments)
        if tool is None or built is None:
            text = (notes.unknown_tool if tool is None else notes.bad_arguments).format(
                tool=call.name
            )
            result = ToolResult(error=text)
        else:
            self._telemetry.started(TOOL_STAGE, heading)
            report = turn.control.before_action(
                built,
                round_number=int(turn.spent(STEPS)),
                max_rounds=self._settings.budget.max_steps,
            )
            if report.decision == "abort":
                result = ToolResult(success=True, output=turn.control.reject_message(built))
            else:
                result = await self._invoke(tool.handler, built.params, call.name)
                if result.success:
                    turn.control.record_success(built, result.output)
                else:
                    turn.control.record_refusal(built, result.error or "")
        text = result.output if result.success else result.error or notes.empty
        self._card_tool(heading, call.arguments, text, result.success)
        self._telemetry.audit(
            "answer.tool",
            "allow" if result.success else "deny",
            f"{call.name}: {'ok' if result.success else 'failed'}",
            arguments=call.arguments,
            result=text,
        )
        urls = [str(url) for url in result.data.get(URLS_KEY, [])]
        # Only what a tool actually returned can ground a claim; an error or a refusal cannot.
        chunk = (
            Chunk(content=text, kind="tool_result", source=urls[0] if urls else call.name)
            if result.success
            else None
        )
        return Message(role="tool", content=text, tool_call_id=call.id), urls, chunk

    def _card_tool(self, heading: str, arguments: str, text: str, ok: bool) -> None:
        notes, limit = self._settings.messages, get_agent_settings().tui.clip_chars
        sections = {
            notes.section_arguments: [arguments],
            notes.section_result: [clip(line, limit) for line in text.splitlines()[:8]]
            or [notes.no_results],
        }
        self._telemetry.card(TOOL_STAGE, heading, sections, status="completed" if ok else "failed")

    async def _invoke(self, handler: Any, params: Mapping[str, Any], name: str) -> ToolResult:
        try:
            outcome = handler(**params)
            return await outcome if inspect.isawaitable(outcome) else outcome  # type: ignore[no-any-return]
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            self._logger.warning("Answer tool failed", tool=name, error=detail)
            return ToolResult(
                error=self._settings.messages.tool_failed.format(tool=name, error=detail)
            )

    def after_act(self, state: AnswerState, config: RunnableConfig) -> str:
        """Back to the model while any budget remains; an exhausted one ends the loop."""
        budget = _turn(config).budget
        spent = budget.exhausted_counters()
        return "finalize" if budget.expired() or STEPS in spent or TOKENS in spent else "reason"

    # -- ground ---------------------------------------------------------------------------

    async def ground(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        """Check an answer that rests on tool results against those results.

        An answer given without a tool call rests on the model's own knowledge, which nothing
        here can check; the prompt asks it to use a tool for anything it is unsure of.
        """
        if self._grounding is None or not state["evidence"] or not state["text"]:
            return {}
        turn = _turn(config)
        outcome = await self._grounding.check(
            turn.question, state["text"], state["evidence"], self._reviser(turn, state["evidence"])
        )
        self._card_grounding(outcome)
        return {"text": outcome.text}

    def _reviser(self, turn: Turn, evidence: list[Chunk]) -> Any:
        """The model's one rewrite of an answer that holds unsupported claims, from the sources."""

        async def revise(candidate: str, unsupported: list[str]) -> str:
            payload = {
                "answer": candidate,
                "unsupported": unsupported,
                "sources": [chunk.content for chunk in evidence],
            }
            result = await self._chat(
                turn,
                [
                    Message(role="system", content=prompt.REVISE_INSTRUCTIONS),
                    Message(role="user", content=json.dumps(payload)),
                ],
                None,
                REVISE_PURPOSE,
            )
            return result.content.strip() or candidate

        return revise

    def _card_grounding(self, outcome: GroundingOutcome) -> None:
        notes = self._settings.messages
        heading = (
            notes.grounded_heading.format(supported=outcome.supported, claims=outcome.claims)
            if outcome.clean
            else notes.ungrounded_heading.format(removed=len(outcome.removed))
        )
        sections = {notes.section_unsupported: outcome.removed} if outcome.removed else {}
        self._telemetry.card(
            "hallucination_check",
            heading,
            sections,
            status="completed" if outcome.clean else "warning",
        )

    # -- finalize -------------------------------------------------------------------------

    async def finalize(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        turn, notes = _turn(config), self._settings.messages
        steps, seconds = int(turn.spent(STEPS)), turn.budget.elapsed()
        if state["failure"]:
            text = notes.failed.format(cause=state["failure"], step=steps, seconds=seconds)
            self._card_final(text, steps, turn, failed=True)
            return {"text": text}
        text = state["text"] or notes.empty
        if state["sources"] and not _URL.search(text):
            unique = list(dict.fromkeys(state["sources"]))
            text = f"{text}\n\n{notes.sources_heading}\n" + "\n".join(f"- {url}" for url in unique)
        stop = state["stop"] or (turn.budget.exhausted_counters() or [""])[0]
        if stop and (not state["text"] or turn.spent(TOOL_CALLS) > 0):
            text = f"{text}\n\n{notes.cap_reached.format(counter=stop)}"
        self._card_final(text, steps, turn, failed=False)
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

    def _card_final(self, text: str, steps: int, turn: Turn, failed: bool) -> None:
        notes = self._settings.messages
        heading = (
            text.splitlines()[0]
            if failed
            else notes.answered.format(
                steps=steps,
                calls=int(turn.spent(TOOL_CALLS)),
                tokens=f"{int(turn.spent(TOKENS)):,}",
                seconds=turn.budget.elapsed(),
            )
        )
        self._telemetry.card(
            STAGE, heading, {notes.section_answer: text.splitlines()},
            status="failed" if failed else "completed",
        )  # fmt: skip


def build_answer_graph(nodes: AnswerNodes) -> Any:
    """Compile the loop. No checkpointer: a single turn has nothing to resume."""
    builder = StateGraph(AnswerState)
    builder.add_node("reason", nodes.reason)
    builder.add_node("act", nodes.act)
    builder.add_node("ground", nodes.ground)
    builder.add_node("finalize", nodes.finalize)
    builder.add_edge(START, "reason")
    builder.add_conditional_edges("reason", nodes.after_reason, ["act", "ground", "finalize"])
    builder.add_edge("ground", "finalize")
    builder.add_conditional_edges("act", nodes.after_act, ["reason", "finalize"])
    builder.add_edge("finalize", END)
    return builder.compile()


def initial_state(system: str, question: str, history: list[Message]) -> AnswerState:
    return {
        "messages": [
            Message(role="system", content=system),
            *history,
            Message(role="user", content=question),
        ],
        "sources": [],
        "evidence": [],
        "text": "",
        "stop": "",
        "failure": "",
    }


__all__ = ["STEPS", "TOOL_CALLS", "AnswerNodes", "Turn", "build_answer_graph", "initial_state"]
