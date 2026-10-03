"""One question in, one answer out: a budgeted ReAct graph, with no scope, plan or audit pipeline.

The graph is `answer_graph.py` over `answer_nodes.py`; this is its facade. It builds the
per-question budget (steps, tool calls, tokens, seconds — all from config), assembles what the
session adds to the question, runs the loop, and never lets an unexpected error leave the shell
waiting: whatever happens, the user gets a reply that says what happened.
"""

from datetime import datetime
from typing import Any

from agent_platform.control_loop import BudgetGuard
from agent_platform.control_loop.types import ControlLoopReport
from agent_platform.control_loop.watchdogs import ToolControlLoop
from agent_platform.logging import describe_error, get_logger, instrument
from agent_platform.observability.loop_reporter import LoopReporter
from agent_platform.tools.registry import NamespacedToolRegistry
from research_agent.config.settings import AnswerSettings
from research_agent.context.session import AnswerContext, SessionContext
from research_agent.domain.budget import TOKENS
from research_agent.graph.answer_graph import build_answer_graph
from research_agent.graph.answer_nodes import LOOP_STAGES, AnswerNodes
from research_agent.graph.answer_state import STEPS, TOOL_CALLS, TURN_KEY, Turn, initial_state
from research_agent.guardrails.answer_grounding import AnswerGrounding
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import answer as prompt
from research_agent.research.cards import cap_reached_card
from research_agent.research.telemetry import RunTelemetry


@instrument("research_agent")
class DirectAnswerer:
    """Answers a plain question, calling tools when the model asks for them."""

    def __init__(
        self,
        caller: StructuredCaller,
        registry: NamespacedToolRegistry,
        settings: AnswerSettings,
        telemetry: RunTelemetry,
        grounding: AnswerGrounding | None = None,
        context: AnswerContext | None = None,
    ) -> None:
        self._context = context
        self._settings = settings
        self._telemetry = telemetry
        self._graph = build_answer_graph(
            AnswerNodes(caller, registry, settings, telemetry, grounding)
        )
        self._logger = get_logger("research_agent")

    async def answer(self, question: str, session_id: str = "") -> str:
        context = (
            await self._context.assemble(session_id, question)
            if self._context
            else SessionContext()
        )
        turn = self._turn(question, context)
        today = datetime.now().astimezone().strftime(self._settings.date_format)
        system = "\n\n".join([prompt.INSTRUCTIONS.format(today=today), *context.blocks])
        try:
            final = await self._graph.ainvoke(
                initial_state(system, question, context.history),
                config={"configurable": {TURN_KEY: turn}},
            )
        except Exception as exc:
            cause = describe_error(exc)
            self._logger.error("Direct answer crashed", cause=cause, question=question)
            self._telemetry.audit("answer.failed", "deny", cause, question=question)
            return self._settings.messages.failed.format(
                cause=cause, step=int(turn.spent(STEPS)), seconds=turn.budget.elapsed()
            )
        text: str = final["text"]
        if not final["failure"] and self._context is not None:
            self._context.remember(session_id, question, self._without_sources(text))
        return text

    def _turn(self, question: str, context: SessionContext) -> Turn:
        loop = LoopReporter(
            self._telemetry.reporter,
            max_steps=self._settings.budget.max_steps,
            stages=LOOP_STAGES,
            detail=self._settings.messages.loop_step_detail,
        )
        return Turn(question, self._budget(), ToolControlLoop(), context, loop)

    def _without_sources(self, text: str) -> str:
        """The answer as the model should remember it: its words, not the citation list.

        A "Sources" list in the history reads as something to go and fetch, so a follow-up
        opens an API URL instead of asking the tool that produced the answer.
        """
        heading = self._settings.messages.sources_heading
        return text.partition(heading)[0].rstrip() or text

    def _budget(self) -> BudgetGuard:
        caps = self._settings.budget
        return BudgetGuard(
            {STEPS: caps.max_steps, TOOL_CALLS: caps.max_tool_calls, TOKENS: caps.max_total_tokens},
            max_seconds=caps.max_seconds,
            on_exhausted=self._announce,
        )

    def _announce(self, report: ControlLoopReport) -> None:
        observed: dict[str, Any] = report.checks[0].observed if report.checks else {}
        self._telemetry.card(
            "budget_tick", *cap_reached_card(report.reason, observed), status="warning"
        )


__all__ = ["DirectAnswerer"]
