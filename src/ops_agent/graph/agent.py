"""The ops agent: one question in, an answer grounded in the queries it ran out.

`handle(query, session_id)` is the shell's contract. A turn is traced (one trace
id for its log lines, facts and audit entries), screened (when the
`prompt_injection` feature is declared), answered by the graph, and recorded as
the outcome of its task (the bracket writes the turn and task facts). The MCP server is connected on the first turn, not at build time:
building an agent must stay offline, and a server that is down should cost one
turn an explanation, not the console its start-up.
"""

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.context import ContextWindow, Task
from agent_platform.context.types import AssembledContext
from agent_platform.knowledge.mcp import McpToolSource
from agent_platform.llm.prompt import TurnLayout
from agent_platform.llm.types import Message
from agent_platform.logging import get_current_trace_id, get_logger, instrument
from agent_platform.memory.procedural.skills import (
    Activation,
    AgentInstructions,
    SkillCatalog,
    build_agent_instructions,
)
from agent_platform.memory.working import WorkingMemoryStore
from agent_platform.observability import StageReporter
from agent_platform.observability.langsmith import NullTraceExporter, TraceExporter, turn_trace
from agent_platform.observability.units import close_task
from agent_platform.security import MessageScreener, ScreenSubject
from agent_platform.security.pii.active import admit_if_on
from agent_platform.tools import NamespacedToolRegistry
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.graph.nodes import OpsNodes
from ops_agent.graph.state import OpsState, initial_state

# How a question ended, as its summary card and its task outcome name it. Each is
# declared, with its flags, under `analytics.tasks.outcomes` in this agent's config.
ANSWERED, STOPPED, BLOCKED, UNAVAILABLE = "answered", "stopped", "blocked", "unavailable"
OUTCOMES = (ANSWERED, STOPPED, BLOCKED, UNAVAILABLE)


class TurnResult(BaseModel):
    """What the shell shows: the reply (and no follow-up choices)."""

    response: str
    outcome: str
    choices: list[str] = Field(default_factory=list)


@instrument("ops_agent")
class OpsAgent:
    """Answers questions about the agents from the analytics source, over MCP."""

    def __init__(
        self,
        graph: Any,
        nodes: OpsNodes,
        registry: NamespacedToolRegistry,
        sources: list[McpToolSource],
        settings: OpsAgentSettings,
        reporter: StageReporter | None = None,
        screen: MessageScreener | None = None,
        trace_exporter: TraceExporter | None = None,
        instructions: AgentInstructions | None = None,
        window: ContextWindow | None = None,
        working_memory: WorkingMemoryStore | None = None,
    ) -> None:
        self._graph = graph
        self._nodes = nodes
        self._registry = registry
        self._sources = sources
        self._settings = settings
        self._reporter = reporter
        self._screen = screen
        self._exporter = trace_exporter or NullTraceExporter()
        # The profile always; skills when procedural memory was built.
        self._instructions = instructions or build_agent_instructions(
            get_settings().skills, store=None, audit=None, reporter=None
        )
        self._history: dict[str, list[Message]] = {}
        # With `context` declared, the turns live in working memory and reach the model
        # through the window; without it, as the conversation they were.
        self._window = window
        self._working = working_memory
        self._connected = False
        self._logger = get_logger("ops_agent")

    @property
    def skill_catalog(self) -> SkillCatalog | None:
        """The skills `/skills` lists (None when procedural memory is withheld)."""
        return self._instructions.catalog

    async def handle(self, query: str, session_id: str) -> TurnResult:
        """One turn, under a trace of its own; the message enters as tokens."""
        query = admit_if_on(query).text
        with turn_trace(
            self._exporter,
            agent=get_settings().agent.id,
            session=session_id,
            inputs={"query": query},
        ) as trace:
            result = await self._summarised(query, session_id)
            # One question is one (implicit) task; its outcome is how the turn ended.
            close_task(result.outcome)
            trace.finish({"response": result.response, "outcome": result.outcome})
            return result

    async def close(self) -> None:
        """Close every MCP session (and end the servers this agent launched)."""
        for source in self._sources:
            closer = getattr(source, "close", None)
            if closer is not None:
                await closer()

    async def _summarised(self, query: str, session_id: str) -> TurnResult:
        if self._reporter is not None:
            self._reporter.begin_turn(session=session_id)
        try:
            result = await self._turn(query, session_id)
        except Exception:
            if self._reporter is not None:
                self._reporter.end_turn("failed", UNAVAILABLE)
            raise
        if self._reporter is not None:
            self._reporter.end_turn("completed", result.outcome)
        return result

    async def _turn(self, query: str, session_id: str) -> TurnResult:
        words = self._settings.words
        refusal = await self._screened(query, session_id)
        if refusal is not None:
            return TurnResult(response=refusal, outcome=BLOCKED)
        problem = await self._connect()
        if problem is not None:
            return TurnResult(response=words.no_tools.format(error=problem), outcome=UNAVAILABLE)
        messages, layout = await self._opening(query, session_id)
        state: OpsState = await self._graph.ainvoke(
            initial_state(session_id, messages, layout, query)
        )
        self._remember(session_id, query, state["answer"])
        verdict, failed = state.get("verdict"), bool(state.get("failure"))
        stopped = verdict is not None and verdict.stops
        outcome = UNAVAILABLE if failed else STOPPED if stopped else ANSWERED
        return TurnResult(response=state["response"], outcome=outcome)

    async def _screened(self, query: str, session_id: str) -> str | None:
        """The refusal, when a declared screen stops the message; None when it may go on."""
        if self._screen is None:
            return None
        subject = ScreenSubject(session_id=session_id, trace_id=get_current_trace_id() or "")
        # Both hosted screens at once; injection still decides first (`screen_both`).
        outcome = (await self._screen.screen_both(query, self._reporter, subject=subject)).stopping
        if outcome is None:
            return None
        return outcome.refusal or self._settings.words.blocked.format(reason=outcome.reason)

    async def _connect(self) -> str | None:
        """List every server's tools into the registry — once, and again for a server that
        said its tools changed; the error if a server is down. The MCP SERVER card, and every
        other protocol event, is the platform's (`mcp/client/events.py`)."""
        try:
            for source in self._sources:
                if not self._connected or source.stale():
                    self._registry.register_all(await source.tools())
        except Exception as exc:  # noqa: BLE001 - said to the operator, retried next turn
            self._logger.warning("MCP tools unavailable", error=repr(exc))
            return str(exc) or repr(exc)
        self._connected = True
        return None

    async def _opening(self, query: str, session_id: str) -> tuple[list[Message], TurnLayout]:
        """The turn's first messages: profile and skill, history, the question, prefetches.

        The skill is chosen first, without a model call, so its prefetch reads (the
        metric catalogue, the view list) are already answered when the model starts.
        """
        activation = self._instructions.select(frozenset(), query)
        planned = _planned(activation)
        # The card first — the skill, what it reads first, and what `propose` will offer
        # (every tool the server listed) — so no read appears without the skill behind it.
        offered = [tool.model_name for tool in self._registry.all_tools()]
        reads = [name for _, name in planned if self._nodes.can_prefetch(name)]
        self._instructions.announce(activation, prefetched=reads, offered=offered)
        self._nodes.begin_turn()
        prefetched, _ran = await self._nodes.prefetch(planned, session_id)
        windowed = await self._windowed(query, session_id)
        history = [] if windowed is not None else self._history.get(session_id, [])
        layout = TurnLayout(
            tuple(self._instructions.parts(activation)),
            history=len(history),
            window_text=windowed.text if windowed is not None else "",
            window_chunks=tuple(windowed.chunks) if windowed is not None else (),
            window_notes=tuple(windowed.applied_strategies) if windowed is not None else (),
        )
        messages = [
            layout.build([]).messages[0],
            *history,
            Message(role="user", content=query),
            *prefetched,
        ]
        return messages, layout

    async def _windowed(self, query: str, session_id: str) -> AssembledContext | None:
        """The session's assembled window, when this agent declares context assembly."""
        if self._window is None:
            return None
        result = await self._window.build(Task(query=query, session_id=session_id), self._reporter)
        return result.window if result.text else None

    def _remember(self, session_id: str, query: str, answer: str) -> None:
        if self._working is not None:
            self._working.get(session_id).add_history(query, answer)
        turns = self._settings.history_turns
        kept = [*self._history.get(session_id, []), Message(role="user", content=query),
                Message(role="assistant", content=answer)]  # fmt: skip
        self._history[session_id] = kept[-2 * turns :] if turns else []


__all__ = ["ANSWERED", "BLOCKED", "OUTCOMES", "STOPPED", "UNAVAILABLE", "OpsAgent", "TurnResult"]


def _planned(activation: Activation) -> list[tuple[str, str]]:
    """(skill, tool) for every prefetch read, each tool once, in skill order."""
    seen: dict[str, str] = {}
    for item in activation.skills:
        for name in item.skill.document.prefetch:
            seen.setdefault(name, item.skill.name)
    return [(skill, name) for name, skill in seen.items()]
