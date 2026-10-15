"""The direct answer as a ReAct loop: THOUGHT (one model call) -> ACTION (the tools it asked for)
-> OBSERVATION (what they returned) -> THOUGHT ... then the HALLUCINATION CHECK and the ANSWER.

    START -> reason -> act -> reason ... -> ground -> finalize -> END

`reason` ends the loop by answering; `act` hands the results back. A budget decides every other
exit: when the last step, the tool-call cap, the token cap or the clock is reached the model is
asked once more *without tools*, so it answers from what it has instead of being cut off mid-search.
A model that still asks for a tool on that last step is asked again, then — still nothing — once
from the observations alone, so an answer the tools found is never dropped for an empty reply.
A search for an undated question carries a note when it is pinned to a past year or finds only
old results (`research/recency.py`): no date named means the latest.
`ground` is the HALLUCINATION CHECK: the answer is checked against what the tools returned, and only if that
check refutes a claim is the model asked once to REVISE it. Every step is a card on the live feed,
numbered "Step N", and every payload is in the log and the audit trail.
"""

import json
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent_platform.context.types import Chunk
from agent_platform.gatekeeper.types import AuthorityContext
from agent_platform.hallucination_check import PASSAGE, Observation
from agent_platform.interface.presentation.text_tree import inline_arguments, readable_lines
from agent_platform.llm.embeddings import tokenize
from agent_platform.llm.prompt import PromptManifest, manifest_from_messages
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message, ModelToolCall
from agent_platform.logging import describe_error, get_logger, instrument
from agent_platform.loop_control import LoopVerdict
from agent_platform.observability.llm_fields import (
    cache_fields,
    llm_request_fields,
    llm_response_fields,
)
from agent_platform.observability.step_events import HALLUCINATION_CHECK
from agent_platform.observability.tool_fields import display_tool_name, tool_request_fields
from agent_platform.tools.path import DUPLICATE, INVALID, UNKNOWN, ToolPath, ToolTurn
from agent_platform.tools.registry import NamespacedToolRegistry
from agent_platform.tools.types import ToolResult
from research_agent.config import get_agent_settings
from research_agent.config.settings import AnswerSettings
from research_agent.domain.budget import TOKENS
from research_agent.graph.answer_state import STEPS, TOOL_CALLS, AnswerState, Turn, turn_of
from research_agent.graph.points import as_points
from research_agent.guardrails.answer_grounding import AnswerGrounding, GroundingOutcome
from research_agent.llm.structured import ModelCallTimeoutError, StructuredCaller
from research_agent.observability.telemetry import ACTOR, RunTelemetry
from research_agent.presentation.narration import clip, role_title
from research_agent.prompts import answer as prompt
from research_agent.research.recency import RecencyGuard
from research_agent.tools import (
    HITS_KEY,
    PAGE_KEY,
    PROVIDER_KEY,
    PUBLISHED_KEY,
    QUERY_KEY,
    URLS_KEY,
)

# The loop's cards, in the order a step emits them.
THOUGHT, ACTION, OBSERVATION, REVISE, ANSWER = (
    "thought",
    "action",
    "observation",
    "revise",
    "answer",
)
LOOP_STAGES = frozenset({THOUGHT, ACTION, OBSERVATION})
# The loop's nodes, as LOOP CONTROL names where it goes next (words: `messages.loop_next`).
ACT, GROUND, FINAL, REASON, FINALIZE = "act", "ground", "final", "reason", "finalize"
# Why, when no budget forced it (the platform's `loop_control.because` words).
_TOOL_REQUESTED, _NO_TOOL_CALL, _ROUNDS_REMAIN = "tool_requested", "no_tool_call", "rounds_remain"
_SECONDS = "seconds"
_WALL_CLOCK = "wall_clock"
# A governor verdict's mechanism → this path's limit word (`messages.loop_because`).
_LAYER_LIMITS = {"token_budget": TOKENS, "step_count": STEPS, "tool_budget": TOOL_CALLS}
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
        tools: ToolPath,
        settings: AnswerSettings,
        telemetry: RunTelemetry,
        grounding: AnswerGrounding | None = None,
        recency: RecencyGuard | None = None,
    ) -> None:
        self._tools = tools
        self._grounding = grounding
        self._recency = recency
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
        verdict = turn.governor.begin_step()
        turn.loop.begin(int(turn.spent(STEPS)))
        turn.steps.begin(int(turn.spent(STEPS)))
        limit = self._limit(turn, verdict)
        if limit == _SECONDS:
            # Too little time left for a model call: answer with what there is.
            self._report_loop(turn, next_step=GROUND, because=limit)
            return {"stop": limit}
        if limit:
            self._report_loop(turn, next_step=FINAL, because=limit)
            return await self._final(turn, state, limit)
        try:
            result = await self._think(turn, state["messages"], self._registry.to_openai_tools())
        except Exception as exc:
            return self._failed(turn, exc)
        asked = result.tool_calls or []
        wanted = asked[: self._tool_room(turn, len(asked))]
        if wanted:
            self._report_loop(
                turn, next_step=ACT, because=_TOOL_REQUESTED, asked=len(asked), kept=len(wanted)
            )
            return {
                "messages": [Message(role="assistant", content=result.content, tool_calls=wanted)]
            }
        self._report_loop(turn, next_step=GROUND, because=_NO_TOOL_CALL)
        return {"text": result.content.strip(), "stop": ""}

    async def _final(self, turn: Turn, state: AnswerState, limit: str) -> dict[str, Any]:
        """The last step: an answer now, from what the tools returned — never an empty reply.

        A model offered no tools may still ask for one (it wants one more page to settle a
        figure). That reply has no text, and taking it as the answer threw away everything the
        tools had found. It is asked again, told it is the last step, and then — if it still
        asks for a tool — once more from the observations alone, with no tool history to resume.
        """
        messages = [*state["messages"], Message(role="user", content=prompt.FINAL_STEP)]
        try:
            text = await self._final_text(turn, messages)
            if not text and state["observations"]:
                text = await self._from_observations(turn, state["observations"])
        except Exception as exc:
            return self._failed(turn, exc)
        used = turn.spent(TOOL_CALLS) > 0
        return {"text": text, "stop": limit if used else ""}

    async def _final_text(self, turn: Turn, messages: list[Message]) -> str:
        """Up to `final_attempts` tries for a text reply with no tools offered."""
        for attempt in range(1, self._settings.final_attempts + 1):
            if self._time_left(turn) < self._settings.min_call_seconds:
                return ""
            result = await self._think(turn, messages, None)
            if result.content.strip():
                return result.content.strip()
            self._logger.warning(
                "Last step gave no answer text",
                attempt=attempt,
                asked_for_tools=[call.name for call in result.tool_calls or []],
                question=turn.question,
            )
        return ""

    async def _from_observations(self, turn: Turn, observations: list[Observation]) -> str:
        """The answer from what the tools returned, in a fresh conversation with nothing to resume."""
        if self._time_left(turn) < self._settings.min_call_seconds:
            return ""
        payload = {
            "question": turn.question,
            "today": turn.asked_at.strftime(self._settings.date_format),
            "observations": [{"source": item.source, "text": item.text} for item in observations],
        }
        messages = [
            Message(role="system", content=prompt.FROM_OBSERVATIONS),
            Message(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        result = await self._think(turn, messages, None, opening=False)
        return result.content.strip()

    async def _think(
        self,
        turn: Turn,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        *,
        opening: bool = True,
    ) -> CompletionResult:
        """One reasoning step: its CONTEXT card, then the model call as one THOUGHT card.

        `opening` is False for a request that does not start from the turn's own opening (the
        answer from observations alone), which is then broken down by message role.
        """
        if opening:
            # The layout lays the request out for the provider's prefix cache (`Volatility`).
            built = turn.layout.build(messages, tools)
            messages, manifest = built.messages, built.manifest
        else:
            manifest = manifest_from_messages(messages, tools)
        request = CompletionRequest(messages=messages, tools=tools, manifest=manifest)
        turn.steps.context(request, purpose=ANSWER_PURPOSE)
        self._emit(turn, THOUGHT, self._waiting(), {}, event_kind="model", status="started")
        result = await self._call(
            turn, messages, tools, self._call_timeout(turn), manifest=manifest
        )
        calls = ", ".join(
            self._notes.tool_call.format(
                tool=display_tool_name(c.name), arguments=inline_arguments(c.arguments)
            )
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
        manifest: PromptManifest | None = None,
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
            manifest=manifest,
        )
        turn.governor.after_model(result.usage.total_tokens, result.cost_usd)
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
            **cache_fields(result.usage),
        }

    def _context_fields(self, turn: Turn) -> dict[str, Any]:
        """What the session added to the question, on the first step only, and only if anything."""
        context, notes = turn.context, self._notes
        if not context.turns:
            return {}
        return {"session_context": [notes.context_turns.format(turns=context.turns)]}

    @staticmethod
    def _usage(result: CompletionResult) -> dict[str, Any]:
        return {
            "tokens": f"{result.usage.total_tokens:,}",
            "seconds": result.latency_ms / 1000,
        }

    def _limit(self, turn: Turn, verdict: LoopVerdict) -> str:
        """Why no tool may be called now ("" while tools are still allowed).

        The governor's verdict first (the deadline, a budget, a cycle, a plateau); then this
        path's own rules: the last step and an empty tool budget answer without tools.
        """
        governor = turn.governor
        if verdict.layer == _WALL_CLOCK or self._time_left(turn) < self._settings.min_call_seconds:
            return _SECONDS
        if verdict.stops:
            return _LAYER_LIMITS.get(verdict.layer or "", verdict.reason)
        if governor.remaining(STEPS) <= 1:
            return STEPS
        if governor.remaining(TOOL_CALLS) <= 0:
            return TOOL_CALLS
        return ""

    @staticmethod
    def _tool_room(turn: Turn, asked: int) -> int:
        """How many of the calls the model asked for the tool budget still allows."""
        return int(min(turn.governor.remaining(TOOL_CALLS), asked))

    def _time_left(self, turn: Turn) -> float:
        return turn.governor.time_left()

    def _call_timeout(self, turn: Turn) -> float:
        return turn.governor.call_limits().timeout_seconds

    def _waiting(self) -> str:
        """The pending THOUGHT card's heading, shown while the model call is in flight."""
        return self._notes.thought_waiting.format(role=role_title(self._settings.role))

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
            return FINALIZE
        last = state["messages"][-1]
        return ACT if last.role == "assistant" and last.tool_calls else GROUND

    # -- ACTION and OBSERVATION -------------------------------------------------------------

    async def act(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        turn = turn_of(config)
        replies: list[Message] = []
        sources: list[str] = []
        observations: list[Observation] = []
        pages: list[str] = []
        for call in state["messages"][-1].tool_calls or []:
            reply, result = await self._run(call, turn)
            urls = [str(url) for url in result.data.get(URLS_KEY, [])]
            replies.append(reply)
            sources.extend(urls)
            observations.extend(_observations(call.name, reply.content, result, urls))
            pages.extend(urls if result.data.get(PAGE_KEY) else [])
        self._report_after_act(turn)
        return {
            "messages": replies,
            "sources": sources,
            "observations": observations,
            "pages": pages,
        }

    # -- LOOP CONTROL -----------------------------------------------------------------------------

    def _report_after_act(self, turn: Turn) -> None:
        """Back to the model: whether another step is affordable is the governor's, at `reason`."""
        self._report_loop(turn, next_step=REASON, because=_ROUNDS_REMAIN)

    def _report_loop(
        self, turn: Turn, *, next_step: str, because: str, asked: int = 0, kept: int = 0
    ) -> None:
        """LOOP CONTROL: every budget the loop weighs, its value now, and the step it chose."""
        criteria: dict[str, Any] = {
            "model_requested_tools": next_step == ACT,
            "steps": self._used(turn, STEPS),
            "tool_calls": self._used(turn, TOOL_CALLS),
            "tokens_left": int(turn.governor.remaining(TOKENS)),
            "seconds_left": max(0, round(self._time_left(turn))),
        }
        if asked > kept:
            criteria["dropped_tool_calls"] = asked - kept
        verdict = turn.governor.verdict
        if verdict is None:
            return
        fields = turn.steps.loop_verdict(
            verdict,
            criteria=criteria,
            next_step=self._notes.loop_next.get(next_step, next_step),
            because=because,
            because_words=self._notes.loop_because,
        )
        self._logger.info("Loop control status", **fields)

    def _used(self, turn: Turn, counter: str) -> str:
        caps = self._settings.budget
        cap = caps.max_steps if counter == STEPS else caps.max_tool_calls
        return f"{turn.spent(counter):g} of {cap:g}"

    async def _run(self, call: ModelToolCall, turn: Turn) -> tuple[Message, ToolResult]:
        name = display_tool_name(call.name)
        self._emit(
            turn,
            ACTION,
            self._notes.tool_call.format(tool=name, arguments=inline_arguments(call.arguments)),
            tool_request_fields(call.name, call.arguments),
            event_kind="tool_request",
        )
        # The status line follows the loop while the tool runs, as every agent's does.
        self._emit(turn, OBSERVATION, "", {}, event_kind="tool_result", status="started")
        result = await self._execute(call, turn)
        text = result.output if result.success else result.error or self._notes.empty
        note = self._recency_note(turn, result)
        if note:
            text = f"{text}\n\n{note}"
        self._observe(turn, name, text, result)
        return Message(role="tool", content=text, tool_call_id=call.id), result

    def _recency_note(self, turn: Turn, result: ToolResult) -> str:
        """A search pinned to the past, or with only old results, for an undated question."""
        query = result.data.get(QUERY_KEY)
        if self._recency is None or not result.success or not isinstance(query, str):
            return ""
        today = turn.asked_at.date()
        published = [str(value) for value in result.data.get(PUBLISHED_KEY, [])]
        verdict = self._recency.check(turn.question, query, published, today)
        return self._recency.note(
            verdict, today, turn.asked_at.strftime(self._settings.date_format)
        )

    async def _execute(self, call: ModelToolCall, turn: Turn) -> ToolResult:
        """The tool, through the platform's one tool path; anything short of a run is said.

        The path validates, suppresses a repeat, audits the run (UNGOVERNED: no Gatekeeper)
        and executes under the Executor's deadline, recording where the proposal stopped.
        """
        outcome = await self._tools.run(
            call,
            ToolTurn(
                loop=turn.governor,
                authority=lambda _built: AuthorityContext(actor=ACTOR),
                # The answer tools read: a failed read asked again fails again.
                remember_failures=True,
            ),
        )
        if outcome.stage == UNKNOWN:
            return ToolResult(error=self._notes.unknown_tool.format(tool=call.name))
        if outcome.stage == INVALID:
            return ToolResult(error=self._notes.bad_arguments.format(tool=call.name))
        if outcome.control is not None and outcome.stage == DUPLICATE:
            # Said to the model, but not evidence: a repeat proved nothing new.
            return ToolResult(error=turn.governor.suppression_message(outcome.control))
        if outcome.execution is None:
            return ToolResult(
                error=self._notes.tool_failed.format(tool=call.name, error=outcome.reason)
            )
        result = outcome.execution.result
        if result.success:
            return result
        detail = result.error or outcome.reason
        return result.model_copy(
            update={"error": self._notes.tool_failed.format(tool=call.name, error=detail)}
        )

    def _observe(self, turn: Turn, tool: str, text: str, result: ToolResult) -> None:
        limit = get_agent_settings().tui.clip_chars
        # A tool's JSON answer reads as a tree, never as JSON.
        lines = [clip(line, limit) for line in readable_lines(text) if line.strip()]
        provider = str(result.data.get(PROVIDER_KEY) or "")
        if not result.success:
            heading = self._notes.observation_failed.format(tool=tool, error=clip(text, limit))
        elif provider:
            heading = self._notes.observation_heading_via.format(
                tool=tool, provider=provider, lines=len(lines)
            )
        else:
            heading = self._notes.observation_heading.format(tool=tool, lines=len(lines))
        fields = {
            "tool": tool,
            "result": "success" if result.success else "failed",
            "observation": lines or [self._notes.no_results],
        }
        status = "completed" if result.success else "failed"
        self._emit(turn, OBSERVATION, heading, fields, event_kind="tool_result", status=status)

    def after_act(self, state: AnswerState, config: RunnableConfig) -> str:
        """Back to the model: its next step asks the governor whether it may go on, and the
        answer comes from what there is when it may not."""
        return REASON

    # -- HALLUCINATION CHECK and REVISE -----------------------------------------------------------------

    async def ground(self, state: AnswerState, config: RunnableConfig) -> dict[str, Any]:
        """Check an answer against the content its sources hold.

        The evidence is what the tools returned and the full page of every URL the answer
        cites (`guardrails/page_evidence.py`). An answer with neither rests on the model's own
        knowledge, which nothing here can check; the prompt asks it to use a tool for anything
        it is unsure of.
        """
        if self._grounding is None or not state["text"]:
            return {}
        turn = turn_of(config)
        evidence = await self._grounding.evidence(
            state["text"], state["observations"], state["pages"], self._time_left(turn)
        )
        if not evidence:
            return {}
        self._emit(turn, HALLUCINATION_CHECK, "", {}, event_kind="grounding", status="started")
        outcome = await self._grounding.check(
            turn.question, state["text"], evidence, self._reviser(turn, evidence)
        )
        self._card_check(turn, outcome)
        return {"text": outcome.text, "quality": outcome.quality}

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
        # One sentence per line: a passage repeats the sentences it is made of.
        lines = [chunk for chunk in evidence if not chunk.metadata.get(PASSAGE)]
        chosen: list[str] = []
        for claim in claims:
            words = set(tokenize(claim))
            ranked = sorted(lines, key=lambda c: -len(words & set(tokenize(c.content))))
            chosen.extend(c.content for c in ranked[:per_claim] if c.content not in chosen)
        return chosen

    def _card_check(self, turn: Turn, outcome: GroundingOutcome) -> None:
        tally = str(outcome.report.get("tally", ""))
        notes = self._notes
        heading = (
            notes.ungrounded_heading.format(removed=len(outcome.removed), tally=tally)
            if not outcome.clean
            else notes.marked_heading.format(count=len(outcome.unverified), tally=tally)
            if outcome.unverified
            else notes.grounded_heading.format(tally=tally)
        )
        # Coloured by what was shown: the answer (done), less a detail or with one marked
        # unverified (warning), or nothing of it.
        status = (
            "blocked"
            if outcome.withheld
            else "completed"
            if outcome.clean and not outcome.unverified
            else "warning"
        )
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
        steps, seconds = int(turn.spent(STEPS)), turn.elapsed()
        if state["failure"]:
            text = self._notes.failed.format(cause=state["failure"], step=steps, seconds=seconds)
            self._card_answer(turn, text, failed=True)
            turn.governor.finish("failed")
            return {"text": text}
        text = self._without_invented_sources(turn, state["text"]) or self._notes.empty
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
        stop = state["stop"] or (turn.governor.exhausted_counters() or [""])[0]
        if stop and (not state["text"] or turn.spent(TOOL_CALLS) > 0):
            text = f"{text}\n\n{self._notes.cap_reached.format(counter=stop)}"
        self._card_answer(turn, text, failed=False)
        turn.governor.finish("finalized_on_cap" if stop else "answered")
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

    def _without_invented_sources(self, turn: Turn, text: str) -> str:
        """An answer no tool informed cites nothing: a Sources line it wrote is invented.

        A model answering from its own knowledge still writes "Sources: Web search" when the
        prompt asks for one, which reads as research that never happened.
        """
        if turn.spent(TOOL_CALLS) > 0 or not self._sources.search(text):
            return text
        body = self._sources.split(text, maxsplit=1)[0].rstrip()
        self._logger.warning(
            "Removed a Sources line from an answer no tool informed",
            question=turn.question,
            removed=text[len(body) :].strip(),
        )
        return body

    def _card_answer(self, turn: Turn, text: str, failed: bool) -> None:
        heading = (
            text.splitlines()[0]
            if failed
            else self._notes.answered.format(
                steps=int(turn.spent(STEPS)),
                calls=int(turn.spent(TOOL_CALLS)),
                tokens=f"{int(turn.spent(TOKENS)):,}",
                seconds=turn.elapsed(),
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


def _observations(tool: str, text: str, result: ToolResult, urls: list[str]) -> list[Observation]:
    """What a call returned, as grounding evidence: a search as one observation per hit.

    Only what a tool actually returned can ground a claim; an error or a refusal cannot. A
    search's hits come from different pages, so each is its own source: one page's snippet
    never joins another's into a premise.
    """
    if not result.success:
        return []
    hits = [str(hit) for hit in result.data.get(HITS_KEY, [])]
    if hits and len(hits) == len(urls):
        return [Observation(hit, source=url) for hit, url in zip(hits, urls, strict=True)]
    return [Observation(text, source=urls[0] if urls else tool)]


__all__ = [
    "ANSWER",
    "ACTION",
    "LOOP_STAGES",
    "HALLUCINATION_CHECK",
    "OBSERVATION",
    "THOUGHT",
    "AnswerNodes",
]
