"""The ops agent: one question in, an answer grounded in the queries it ran out.

`handle(query, session_id)` is the shell's contract. A turn is traced (one trace
id for its log lines, facts and audit entries), screened (when the
`prompt_injection` feature is declared), answered by the graph, and recorded as
a turn fact. The MCP server is connected on the first turn, not at build time:
building an agent must stay offline, and a server that is down should cost one
turn an explanation, not the console its start-up.
"""

import time
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.llm.types import Message
from agent_platform.logging import get_current_trace_id, get_logger, instrument
from agent_platform.mcp import McpToolSource
from agent_platform.observability import FactRecorder, NullFactRecorder, StageReporter
from agent_platform.observability.facts import TurnOutcomeFact
from agent_platform.observability.langsmith import NullTraceExporter, TraceExporter, turn_trace
from agent_platform.security import MessageScreener, ScreenSubject
from agent_platform.tools import NamespacedToolRegistry
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.graph.state import OpsState, initial_state

# How a turn ended, as its summary card and its outcome fact name it.
ANSWERED, STOPPED, BLOCKED, UNAVAILABLE = "answered", "stopped", "blocked", "unavailable"


class TurnResult(BaseModel):
    """What the shell shows: the reply (and no follow-up choices)."""

    response: str
    outcome: str
    choices: list[str] = Field(default_factory=list)


@instrument("ops_agent")
class OpsAgent:
    """Answers questions about the agents from the analytics mart, over MCP."""

    def __init__(
        self,
        graph: Any,
        registry: NamespacedToolRegistry,
        sources: list[McpToolSource],
        settings: OpsAgentSettings,
        reporter: StageReporter | None = None,
        screen: MessageScreener | None = None,
        facts: FactRecorder | None = None,
        trace_exporter: TraceExporter | None = None,
    ) -> None:
        self._graph = graph
        self._registry = registry
        self._sources = sources
        self._settings = settings
        self._reporter = reporter
        self._screen = screen
        self._facts = facts or NullFactRecorder()
        self._exporter = trace_exporter or NullTraceExporter()
        self._history: dict[str, list[Message]] = {}
        self._connected = False
        self._logger = get_logger("ops_agent")

    async def handle(self, query: str, session_id: str) -> TurnResult:
        """One turn, under a trace of its own."""
        with turn_trace(
            self._exporter,
            agent=get_settings().agent.id,
            session=session_id,
            inputs={"query": query},
        ) as trace:
            started = time.perf_counter()
            result, state = await self._summarised(query, session_id)
            self._record(result, state, session_id, started)
            trace.finish({"response": result.response, "outcome": result.outcome})
            return result

    async def close(self) -> None:
        """Close every MCP session (and end the servers this agent launched)."""
        for source in self._sources:
            closer = getattr(source, "close", None)
            if closer is not None:
                await closer()

    async def _summarised(self, query: str, session_id: str) -> tuple[TurnResult, OpsState | None]:
        if self._reporter is not None:
            self._reporter.begin_turn(session=session_id)
        try:
            result, state = await self._turn(query, session_id)
        except Exception:
            if self._reporter is not None:
                self._reporter.end_turn("failed", UNAVAILABLE)
            raise
        if self._reporter is not None:
            self._reporter.end_turn("completed", result.outcome)
        return result, state

    async def _turn(self, query: str, session_id: str) -> tuple[TurnResult, OpsState | None]:
        words = self._settings.words
        refusal = await self._screened(query, session_id)
        if refusal is not None:
            return TurnResult(response=refusal, outcome=BLOCKED), None
        problem = await self._connect()
        if problem is not None:
            return TurnResult(
                response=words.no_tools.format(error=problem), outcome=UNAVAILABLE
            ), None
        state: OpsState = await self._graph.ainvoke(
            initial_state(session_id, self._messages(query, session_id))
        )
        self._remember(session_id, query, state["answer"])
        stopped = bool(state["messages"][-1].tool_calls)
        return TurnResult(
            response=state["response"], outcome=STOPPED if stopped else ANSWERED
        ), state

    async def _screened(self, query: str, session_id: str) -> str | None:
        """The refusal, when a declared screen stops the message; None when it may go on."""
        if self._screen is None:
            return None
        subject = ScreenSubject(session_id=session_id, trace_id=get_current_trace_id() or "")
        outcome = await self._screen.injection(query, self._reporter, subject=subject)
        if not outcome.blocked:
            return None
        return outcome.refusal or self._settings.words.blocked.format(reason=outcome.reason)

    async def _connect(self) -> str | None:
        """List every server's tools into the registry, once; the error if a server is down."""
        if self._connected:
            return None
        try:
            for source in self._sources:
                self._registry.register_all(await source.tools())
        except Exception as exc:  # noqa: BLE001 - said to the operator, retried next turn
            self._logger.warning("MCP tools unavailable", error=repr(exc))
            return str(exc) or repr(exc)
        self._connected = True
        return None

    def _messages(self, query: str, session_id: str) -> list[Message]:
        return [
            Message(role="system", content=self._settings.system_prompt),
            *self._history.get(session_id, []),
            Message(role="user", content=query),
        ]

    def _remember(self, session_id: str, query: str, answer: str) -> None:
        turns = self._settings.history_turns
        kept = [*self._history.get(session_id, []), Message(role="user", content=query),
                Message(role="assistant", content=answer)]  # fmt: skip
        self._history[session_id] = kept[-2 * turns :] if turns else []

    def _record(
        self, result: TurnResult, state: OpsState | None, session_id: str, started: float
    ) -> None:
        latency = (time.perf_counter() - started) * 1000
        if state is None:
            fact = TurnOutcomeFact(
                session_id=session_id, outcome=result.outcome, latency_ms=latency, answered=False
            )
        else:
            fact = TurnOutcomeFact(
                session_id=session_id,
                outcome=result.outcome,
                latency_ms=latency,
                answered=result.outcome == ANSWERED,
                loop_aborted=result.outcome == STOPPED,
                tool_calls=state["tool_calls"],
                tool_failures=state["tool_failures"],
                unverified_claims=state["unverified"],
            )
        self._facts.record(fact)


__all__ = ["ANSWERED", "BLOCKED", "STOPPED", "UNAVAILABLE", "OpsAgent", "TurnResult"]
