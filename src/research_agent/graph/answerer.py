"""One question in, one answer out: a budgeted ReAct graph, with no scope, plan or audit pipeline.

The graph is `answer_graph.py` over `answer_nodes.py`; this is its facade. It builds the
per-question budget (steps, tool calls, tokens, seconds — all from config), assembles what the
session adds to the question, runs the loop, and never lets an unexpected error leave the shell
waiting: whatever happens, the user gets a reply that says what happened.
"""

from datetime import datetime

from pydantic import BaseModel

from agent_platform.llm.prompt import INSTRUCTIONS, PromptPart, TurnLayout
from agent_platform.logging import describe_error, get_logger, instrument
from agent_platform.loop_control import LoopBudget, LoopGovernor, build_loop_governor
from agent_platform.memory.procedural.skills import AgentInstructions
from agent_platform.observability.loop_reporter import LoopReporter
from agent_platform.observability.step_events import StepEvents
from agent_platform.tools.path import ToolPath
from agent_platform.tools.registry import NamespacedToolRegistry
from research_agent.config import get_agent_settings
from research_agent.config.settings import AnswerSettings
from research_agent.context.session import AnswerContext, SessionContext
from research_agent.graph.answer_graph import build_answer_graph
from research_agent.graph.answer_nodes import LOOP_STAGES, AnswerNodes
from research_agent.graph.answer_state import STEPS, TURN_KEY, Turn, initial_state
from research_agent.guardrails.answer_grounding import AnswerGrounding
from research_agent.llm.structured import StructuredCaller
from research_agent.observability.telemetry import RunTelemetry
from research_agent.prompts import answer as prompt
from research_agent.research.recency import RecencyGuard

# The session's own notes (its state) when no context window carries them.
SESSION = "session"
TODAY = "today"


class DirectAnswer(BaseModel):
    """A direct answer's text, and whether the loop failed to produce a real one."""

    text: str
    failed: bool = False


@instrument("research_agent")
class DirectAnswerer:
    """Answers a plain question, calling tools when the model asks for them."""

    def __init__(
        self,
        caller: StructuredCaller,
        registry: NamespacedToolRegistry,
        tools: ToolPath,
        settings: AnswerSettings,
        telemetry: RunTelemetry,
        grounding: AnswerGrounding | None = None,
        context: AnswerContext | None = None,
        recency: RecencyGuard | None = None,
        instructions: AgentInstructions | None = None,
    ) -> None:
        self._context = context
        self._instructions = instructions
        self._settings = settings
        self._telemetry = telemetry
        self._graph = build_answer_graph(
            AnswerNodes(caller, registry, tools, settings, telemetry, grounding, recency)
        )
        self._logger = get_logger("research_agent")

    async def answer(self, question: str, session_id: str = "") -> str:
        """The answer's text (what the shell shows)."""
        return (await self.respond(question, session_id)).text

    async def respond(self, question: str, session_id: str = "") -> DirectAnswer:
        """The answer, and whether the loop failed to produce one."""
        context = (
            await self._context.assemble(session_id, question)
            if self._context
            else SessionContext()
        )
        turn = self._turn(question, context)
        system = turn.layout.build([]).messages[0].content
        try:
            final = await self._graph.ainvoke(
                initial_state(system, question, context.history),
                config={"configurable": {TURN_KEY: turn}},
            )
        except Exception as exc:
            cause = describe_error(exc)
            self._logger.error("Direct answer crashed", cause=cause, question=question)
            self._telemetry.audit("answer.failed", "deny", cause, question=question)
            failed = self._settings.messages.failed.format(
                cause=cause, step=int(turn.spent(STEPS)), seconds=turn.elapsed()
            )
            return DirectAnswer(text=failed, failed=True)
        text: str = final["text"]
        quality = final.get("quality")
        if quality is not None and self._context is not None:
            self._context.observe_quality(session_id, quality)
        if not final["failure"] and self._context is not None:
            self._context.remember(session_id, question, self._without_sources(text))
        return DirectAnswer(text=text, failed=bool(final["failure"]))

    def _profile(self, question: str) -> list[PromptPart]:
        """The profile and the answerer's skills, above its own instructions."""
        if self._instructions is None:
            return []
        signal = get_agent_settings().instructions.role_signal.format(role=prompt.ROLE)
        return self._instructions.parts(
            self._instructions.compose(frozenset({signal}), question).activation
        )

    def _layout(self, question: str, context: SessionContext, asked_at: datetime) -> TurnLayout:
        """The opening every step's request shares: profile, instructions, session, window."""
        today = asked_at.strftime(self._settings.date_format)
        parts = [*self._profile(question), PromptPart(INSTRUCTIONS, prompt.INSTRUCTIONS)]
        tail = [
            *(PromptPart(SESSION, block) for block in context.blocks),
            PromptPart(TODAY, prompt.TODAY.format(today=today)),
        ]
        window = context.window
        return TurnLayout(
            instructions=tuple(parts),
            tail=tuple(tail),
            history=len(context.history),
            window_text=window.text if window is not None else "",
            window_chunks=tuple(window.chunks) if window is not None else (),
            window_notes=tuple(window.applied_strategies) if window is not None else (),
        )

    def _turn(self, question: str, context: SessionContext) -> Turn:
        loop = LoopReporter(
            self._telemetry.reporter,
            max_steps=self._settings.budget.max_steps,
            stages=LOOP_STAGES,
            detail=self._settings.messages.loop_step_detail,
        )
        steps = StepEvents(self._telemetry.reporter, max_steps=self._settings.budget.max_steps)
        asked_at = datetime.now().astimezone()
        layout = self._layout(question, context, asked_at)
        governor = self._governor()
        governor.begin_turn()
        return Turn(question, governor, context, loop, steps, asked_at, layout)

    def _without_sources(self, text: str) -> str:
        """The answer as the model should remember it: its words, not the citation list.

        A "Sources" list in the history reads as something to go and fetch, so a follow-up
        opens an API URL instead of asking the tool that produced the answer.
        """
        heading = self._settings.messages.sources_heading
        return text.partition(heading)[0].rstrip() or text

    def _governor(self) -> LoopGovernor:
        """This question's governor, over the answer path's own budget."""
        caps = self._settings.budget
        return build_loop_governor(
            LoopBudget(
                max_steps=caps.max_steps,
                max_tool_calls=caps.max_tool_calls,
                max_tokens=caps.max_total_tokens,
                max_seconds=caps.max_seconds,
                call_timeout_seconds=self._settings.call_timeout_seconds,
            )
        )


__all__ = ["DirectAnswer", "DirectAnswerer"]
