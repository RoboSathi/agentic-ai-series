"""The ops agent's nodes: propose ⇄ execute, then ground.

The loop is typed tool use, not ReAct: the model proposes validated tool calls,
there is no thought step. Every card is the platform's (`StepEvents`), so a step
here reads exactly as it does on the support agent.

propose  the model reads the conversation and either asks for tools or answers
         (LLM REQUEST, LLM RESPONSE, then LOOP CONTROL)
execute  each tool call runs the governed way (`GovernedCalls`): TOOL REQUEST,
         GATEKEEPER, EXECUTOR, MCP CALL, TOOL RESULT; results go back
ground   the hallucination check — level 1, token confidence (the platform's);
         level 2, every figure found in a query result (this agent's); the gate cuts
         a guessed sentence — then the reply: the answer, any unverified figures,
         the SQL and its rows
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.gatekeeper.types import AuthorityDecision
from agent_platform.hallucination_check.card import log_grounding
from agent_platform.llm.client import LLMClient
from agent_platform.llm.prompt import TurnLayout
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message, ModelToolCall
from agent_platform.logging import describe_error, get_logger, instrument
from agent_platform.loop_control import (
    TOOL_CALLS,
    LoopDecision,
    LoopGovernor,
    LoopVerdict,
    route_on_verdict,
)
from agent_platform.observability import StageReporter
from agent_platform.observability.step_events import StepEvents
from agent_platform.observability.tool_fields import display_tool_name
from agent_platform.security.prompt_injection.detector import InstructionDataSeparator
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.types import ToolCall
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.domain import QueryRecord
from ops_agent.graph.state import OpsState
from ops_agent.graph.tool_calls import CallOutcome, GovernedCalls
from ops_agent.guardrails import OpsGrounding, OpsGroundingResult, SqlReviewer
from ops_agent.presentation import present
from ops_agent.presentation.cards import (
    REFLECT_STAGE,
    deterministic_card,
    result_card,
    review_card,
)

PROPOSE, EXECUTE, REVIEW, GROUND = "propose", "execute", "review", "ground"


class AnswerCutOffError(RuntimeError):
    """The model's answer was cut off at the token cap, even after a retry with a larger one."""


# Why the loop chose its next step (wording: `observability.step_events.loop_control`).
_TOOL_REQUESTED, _NO_TOOL_CALL = "tool_requested", "no_tool_call"
# A verdict that stops the loop sends the turn straight to the answer.
_STOPPED: dict[LoopDecision, str] = {"finalize": GROUND, "abort": GROUND, "pause": GROUND}
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
        grounding: OpsGrounding | None = None,
        reviewer: SqlReviewer | None = None,
        separator: InstructionDataSeparator | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._calls = calls
        self._settings = settings
        self._reporter = reporter
        self._steps = StepEvents(reporter, max_steps=settings.max_steps, model=llm.model)
        # The hallucination check: token confidence (level 1), the figures (level 2), the gate.
        self._grounding = grounding
        # Reflection on the SQL the model wrote, before the answer is grounded (None: withheld).
        self._reviewer = reviewer
        # Tool results are data, never instructions: delimited before the model reads them
        # (with `prompt_injection`; None when that feature is withheld).
        self._separator = separator
        self._logger = get_logger("ops_agent")
        self._guard_log = get_logger("guardrails")

    async def propose(self, state: OpsState) -> dict[str, Any]:
        """One step: the model reads the conversation and asks for tools, or answers."""
        step = state["steps"] + 1
        self._steps.begin(step)
        governor = self._calls.governor
        verdict = governor.begin_step()
        if verdict.stops:
            # The step cap, the deadline, a budget, a cycle or a plateau: answer now.
            self._report_loop(verdict, wants_tools=False)
            return {"steps": step, "verdict": verdict}
        try:
            result = await self._ask(state, governor)
        except Exception as exc:  # noqa: BLE001 - said to the person; the turn ends failed
            error = describe_error(exc)
            self._logger.error("Model call failed", error=error, step=step)
            return {"steps": step, "verdict": verdict, "failure": error}
        governor.after_model(result.usage.total_tokens, result.cost_usd)
        wants_tools = bool(result.tool_calls)
        self._report_loop(verdict, wants_tools=wants_tools)
        reply = Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
        return {"messages": [reply], "steps": step, "answer": result.content, "verdict": verdict}

    async def _ask(self, state: OpsState, governor: LoopGovernor) -> CompletionResult:
        """The step's model call, capped by what the token budget has left and cancelled at
        the deadline."""
        # A spent tool budget: no tools offered, and the model is told to answer from what it has
        # (offering them again only gets every call stopped).
        spent = governor.remaining(TOOL_CALLS) <= 0
        tools = [] if spent else self._registry.to_openai_tools()
        messages = [*state["messages"], *self._budget_note(spent)]
        built = state.get("layout", TurnLayout()).build(messages, tools)
        purpose = get_settings().observability.llm_purposes.next_action
        result = await self._complete(built, tools, governor, self._settings.max_tokens, purpose)
        if _cut_off(result, self._settings.cut_off_finish):
            # The model's reasoning used the whole cap before the answer was written: once more,
            # with the larger cap, rather than send a cut-off answer.
            self._logger.warning("Answer cut off at the token cap", cap=self._settings.max_tokens)
            retry = self._settings.retry_max_tokens
            result = await self._complete(built, tools, governor, retry, purpose)
            if _cut_off(result, self._settings.cut_off_finish):
                raise AnswerCutOffError(self._settings.words.cut_off.format(tokens=retry))
        return result

    async def _complete(
        self, built: Any, tools: list[Any], governor: LoopGovernor, cap: int, purpose: str
    ) -> CompletionResult:
        """One model call, capped by what the token budget has left and the deadline."""
        limits = governor.call_limits(cap)
        request = CompletionRequest(
            messages=built.messages,
            tools=tools,
            temperature=self._settings.temperature,
            max_tokens=limits.max_tokens,
            manifest=built.manifest,
        )
        # A step may ask for tools or answer; the response card names which it did.
        self._steps.llm_request(request, purpose=purpose)
        async with governor.deadline(limits.timeout_seconds):
            result = await self._llm.complete(request)
        self._steps.llm_response(result, purpose=purpose)
        return result

    def _budget_note(self, spent: bool) -> list[Message]:
        return [Message(role="user", content=self._settings.words.tools_spent)] if spent else []

    def after_propose(self, state: OpsState) -> str:
        """The governor's verdict first; then tools while the model asks for them; then the answer."""
        if state.get("failure"):
            return GROUND
        stopped = route_on_verdict(state.get("verdict"), _STOPPED, "")
        if stopped:
            return stopped
        last = state["messages"][-1]
        return EXECUTE if last.role == "assistant" and last.tool_calls else GROUND

    @property
    def reviews(self) -> bool:
        """Whether the data review was built (the graph has a `review` node)."""
        return self._reviewer is not None

    def after_execute(self, state: OpsState) -> str:
        """REFLECT after a step that ran a data query, before the model answers from it."""
        return REVIEW if self._step_data(state) else PROPOSE

    def _step_data(self, state: OpsState) -> list[QueryRecord]:
        """The data queries this step ran (none when the review is withheld)."""
        if self._reviewer is None:
            return []
        return self._reviewer.data(state["queries"][state["reviewed"] :])

    async def review(self, state: OpsState) -> dict[str, Any]:
        """REFLECT on the step's data: the critic reads model-written SQL against the question
        (revise: the model fixes it before answering); catalogue calls are decided in code."""
        assert self._reviewer is not None
        data = self._step_data(state)
        written = self._reviewer.reviewed(data)
        update: dict[str, Any] = {"reviewed": len(state["queries"])}
        self._steps.begin(None)
        if not written or state["reviews"] >= self._settings.sql_review.max_reviews:
            status, heading, card = deterministic_card(
                data, state["question"], capped=bool(written)
            )
            self._steps.step_card(REFLECT_STAGE, status, heading, card)
            return update
        found = await self._reviewer.review(state["question"], written)
        if found.call is not None:
            # The review is a model call of the turn: it counts against the turn's budgets.
            self._calls.governor.after_model(found.call.tokens, found.cost_usd)
        reviews = state["reviews"] + 1
        again = found.revise and reviews < self._settings.sql_review.max_reviews
        status, heading, card = review_card(found, data, state["question"], reviews, again)
        self._steps.step_card(REFLECT_STAGE, status, heading, card)
        if again:
            # The finding goes with the result: the model fixes the query before it answers.
            update["messages"] = [self._reviewer.revise_message(found)]
        return {**update, "reviews": reviews}

    def _report_loop(self, verdict: LoopVerdict, *, wants_tools: bool) -> None:
        """LOOP CONTROL: the governor's verdict, whether the model asked for tools, what follows."""
        if verdict.stops:
            next_step, because = GROUND, verdict.reason_code
        else:
            next_step = EXECUTE if wants_tools else GROUND
            because = _TOOL_REQUESTED if wants_tools else _NO_TOOL_CALL
        fields = self._steps.loop_verdict(
            verdict,
            criteria={"model_requested_tools": wants_tools},
            next_step=next_step,
            because=because,
        )
        self._logger.info("Loop control status", **fields)

    def begin_turn(self) -> None:
        """A new question: the loop forgets the last one (before any prefetch read runs)."""
        self._calls.begin_turn()

    def can_prefetch(self, name: str) -> bool:
        """Whether a skill's read can run before the model: known, read-only, no arguments."""
        tool = self._registry.get(name)
        return tool is not None and not tool.side_effecting and not _required(tool)

    async def prefetch(
        self, planned: list[tuple[str, str]], session_id: str
    ) -> tuple[list[Message], list[str]]:
        """A skill's first reads, run before the model is asked: the messages, and what ran.

        Each is shown and governed like a model's call, and handed to the model as one, so it
        starts where the skill's first steps end. A read that needs arguments, or would
        change something, is skipped.
        """
        messages: list[Message] = []
        ran: list[str] = []
        for skill, name in planned:
            tool = self._registry.get(name)
            if tool is None or not self.can_prefetch(name):
                continue
            call = ModelToolCall(
                id=self._settings.prefetch_call_id.format(tool=tool.model_name),
                name=tool.model_name,
                arguments=_NO_ARGUMENTS,
            )
            self._steps.tool_request(call.name, call.arguments, prefetched_for=skill)
            outcome = await self._calls.run(call, session_id, on_decision=self._decided)
            self._report_call(call.name, outcome)
            messages += [
                Message(role="assistant", content="", tool_calls=[call]),
                Message(
                    role="tool", content=self._as_data(name, outcome.text), tool_call_id=call.id
                ),
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
            outcome = await self._calls.run(call, state["session_id"], on_decision=self._decided)
            self._report_call(call.name, outcome)
            content = self._as_data(call.name, outcome.text)
            replies.append(Message(role="tool", content=content, tool_call_id=call.id))
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
        words = self._settings.words
        verdict = state.get("verdict")
        if state.get("failure"):
            answer, outcome = words.model_failed.format(error=state["failure"]), "failed"
        elif verdict is not None and verdict.stops:
            answer, outcome = words.stopped.format(steps=state["steps"]), "finalized_on_cap"
        else:
            answer, outcome = state["answer"], "answered"
        self._calls.governor.finish(outcome)  # type: ignore[arg-type]
        # Only the model's answer is checked: the agent's own words carry no claim of its.
        checked = self._check(answer, state["queries"]) if outcome == "answered" else None
        sent = checked.response if checked is not None else answer
        unverified = checked.unverified if checked is not None else []
        return {
            "answer": answer,
            "response": present(sent, state["queries"], unverified, self._settings),
            "unverified": None if checked is None else len(unverified),
        }

    def _check(self, answer: str, queries: list[QueryRecord]) -> OpsGroundingResult | None:
        """The hallucination check (none when withheld), reported as every agent's card is."""
        if self._grounding is None:
            return None
        checked = self._grounding.check(answer, queries)
        log_grounding(self._guard_log, checked.card)
        card = checked.card
        self._steps.begin(None)
        self._steps.hallucination_check(
            checked.status, card.outcome if checked.escalated else card.tally, card
        )
        return checked

    def _as_data(self, tool: str, text: str) -> str:
        """A tool's result as the model reads it: inside the data delimiters, never as an
        instruction; an injection phrasing in it is logged (the result is not changed)."""
        if self._separator is None:
            return text
        separated = self._separator.separate("", text)
        if separated.suspicious_phrases:
            self._logger.warning(
                "Injection phrasing in a tool result",
                tool=tool,
                phrases=separated.suspicious_phrases,
            )
        return separated.delimited_data

    def _decided(self, call: ToolCall, decision: AuthorityDecision) -> None:
        """The Gatekeeper's ruling on one call, as it ruled (the audit entry is its own)."""
        self._steps.gatekeeper(call, decision)

    def _report_call(self, name: str, outcome: CallOutcome) -> None:
        """The call's result, named for what came back, with the tool's own preview."""
        preview = (
            result_card(display_tool_name(name), outcome.data, self._settings)
            if outcome.succeeded
            else None
        )
        self._steps.tool_outcome(
            name, outcome.path, outcome.text, refused=outcome.refused, preview=preview
        )


def _cut_off(result: CompletionResult, finish: str) -> bool:
    """An answer (no tool call) the token cap cut short."""
    return result.finish_reason == finish and not result.tool_calls


__all__ = ["EXECUTE", "GROUND", "PROPOSE", "REVIEW", "AnswerCutOffError", "OpsNodes"]


def _required(tool: Any) -> bool:
    """Whether a tool needs arguments a prefetch has no way to know."""
    return any(field.is_required() for field in tool.args_schema.model_fields.values())
