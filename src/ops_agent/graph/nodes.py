"""The ops agent's nodes: reason ⇄ act, then ground.

reason   the model reads the conversation and either asks for tools or answers
act      each tool call runs the governed way (`GovernedCalls`); results go back
ground   the answer's figures are checked against the query results, and the
         reply is assembled: the answer, any unverified figures, the SQL and its rows
"""

from typing import Any

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import get_logger, instrument
from agent_platform.observability import StageReporter
from agent_platform.observability.types import EventKind, StatusKind
from agent_platform.tools import NamespacedToolRegistry
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.domain import QueryRecord
from ops_agent.graph.state import OpsState
from ops_agent.graph.tool_calls import GovernedCalls
from ops_agent.guardrails import FigureCheck
from ops_agent.presentation import present

ACT, GROUND = "act", "ground"


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
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._calls = calls
        self._settings = settings
        self._reporter = reporter
        self._figures = figures
        self._logger = get_logger("ops_agent")

    async def reason(self, state: OpsState) -> dict[str, Any]:
        step = state["steps"] + 1
        cards, stage = self._settings.cards, self._settings.stages.reason
        self._card("started", stage, cards.thinking.format(step=step), "model")
        result = await self._llm.complete(
            CompletionRequest(
                messages=state["messages"],
                tools=self._registry.to_openai_tools(),
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
        )
        self._card("completed", stage, cards.answered.format(step=step), "model")
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
            outcome = await self._calls.run(call, state["session_id"])
            self._report_call(call.name, outcome.succeeded, outcome.refused, outcome.query)
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
        unsupported = self._unsupported(answer, state["queries"])
        return {
            "answer": answer,
            "response": present(answer, state["queries"], unsupported, self._settings),
            "unverified": None if self._figures is None else len(unsupported),
        }

    def _unsupported(self, answer: str, queries: list[QueryRecord]) -> list[str]:
        if self._figures is None:
            return []
        unsupported = self._figures.unsupported(answer, queries)
        cards, stage = self._settings.cards, self._settings.stages.ground
        if unsupported:
            summary = cards.ungrounded.format(count=len(unsupported))
            self._card("warning", stage, summary, "grounding", {"figures": unsupported})
        else:
            self._card("completed", stage, cards.grounded, "grounding")
        return unsupported

    def _report_call(
        self, tool: str, succeeded: bool, refused: bool, query: QueryRecord | None
    ) -> None:
        cards, stage = self._settings.cards, self._settings.stages.act
        if refused:
            self._card("blocked", stage, cards.refused.format(tool=tool), "tool_result")
        elif not succeeded:
            self._card("failed", stage, cards.failed.format(tool=tool), "tool_result")
        else:
            rows = query.row_count if query is not None else 0
            fields = {"sql": query.sql} if query is not None else {}
            summary = cards.called.format(tool=tool, rows=rows)
            self._card("completed", stage, summary, "tool_result", fields)

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


__all__ = ["ACT", "GROUND", "OpsNodes"]
