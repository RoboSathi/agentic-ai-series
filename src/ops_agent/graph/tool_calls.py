"""One tool call, the governed way: the platform's single tool path, read as this agent reads it.

The MCP server's tools are ordinary `ToolDefinition`s, so they take the same
path as any agent's (`agent_platform.tools.path`): the registry validates the
arguments; the turn's loop control suppresses a repeated query; the Gatekeeper
rules on the call against this agent's policy (and writes the audit entry); the
Executor runs it under its deadline. With the Gatekeeper withheld the call still
goes through the Executor, and the audit entry says UNGOVERNED. The path records
where every proposal stopped; this class turns that into what the model reads.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_platform.gatekeeper.types import AuthorityContext, AuthorityDecision
from agent_platform.llm.types import ModelToolCall
from agent_platform.loop_control.types import LoopControlReport
from agent_platform.loop_control.watchdogs import ToolLoopControl
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.path import DUPLICATE, ToolOutcome, ToolPath, ToolTurn
from agent_platform.tools.types import ToolCall, ToolResult
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.domain import QueryRecord


class CallOutcome(BaseModel):
    """What one call told the model, the query it answered (if one), and how far it got."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    succeeded: bool
    refused: bool = False
    query: QueryRecord | None = None
    # Where the platform tool path stopped the call, and everything it learned on the way.
    path: ToolOutcome | None = None
    # The tool's structured result (rows, metrics, views, columns), for the result card.
    data: dict[str, Any] = Field(default_factory=dict)


class GovernedCalls:
    """Runs the model's tool calls through the platform tool path, one loop control a turn."""

    def __init__(
        self,
        registry: NamespacedToolRegistry,
        path: ToolPath,
        settings: OpsAgentSettings,
        actor: str,
    ) -> None:
        self._registry = registry
        self._path = path
        self._settings = settings
        self._actor = actor
        self._loop = ToolLoopControl()

    def begin_turn(self) -> None:
        """A new question: nothing it asks for has been asked before; the step cap is declared."""
        self._loop = ToolLoopControl(step_cap=self._settings.max_steps)

    def check_step(self, step: int, wants_tools: bool) -> LoopControlReport:
        """Whether the turn's step cap allows another round of tools (reported to the bracket)."""
        return self._loop.evaluate_round(
            step + 1 if wants_tools else step, self._settings.max_steps
        )

    async def run(
        self,
        call: ModelToolCall,
        session_id: str,
        step: int,
        on_decision: Callable[[ToolCall, AuthorityDecision], None] | None = None,
    ) -> CallOutcome:
        outcome = await self._path.run(
            call,
            ToolTurn(
                loop=self._loop,
                round_number=step,
                max_rounds=self._settings.max_steps,
                authority=lambda built: self._authority(built, session_id),
                on_decision=on_decision,
            ),
        )
        return self._read(call.name, outcome).model_copy(update={"path": outcome})

    def _authority(self, call: ToolCall, session_id: str) -> AuthorityContext:
        return AuthorityContext(
            actor=self._actor,
            session_id=session_id,
            offered_actions=[tool.qualified_name for tool in self._registry.all_tools()],
        )

    def _read(self, name: str, outcome: ToolOutcome) -> CallOutcome:
        words = self._settings.words
        if outcome.call is None:
            return CallOutcome(text=words.invalid_call.format(tool=name), succeeded=False)
        if outcome.stage == DUPLICATE:
            return CallOutcome(text=self._loop.reject_message(outcome.call), succeeded=False)
        if outcome.execution is None:
            text = words.refused_tool.format(action=outcome.call.action, reason=outcome.reason)
            return CallOutcome(text=text, succeeded=False, refused=True)
        return _outcome(outcome.execution.result)


def _outcome(result: ToolResult) -> CallOutcome:
    if not result.success:
        return CallOutcome(text=result.error or result.output, succeeded=False, data=result.data)
    return CallOutcome(text=result.output, succeeded=True, query=_query(result), data=result.data)


def _query(result: ToolResult) -> QueryRecord | None:
    """The result as a query record, when it is one (a tool that returned rows)."""
    try:
        return QueryRecord.model_validate(result.data)
    except ValidationError:
        return None


__all__ = ["CallOutcome", "GovernedCalls"]
