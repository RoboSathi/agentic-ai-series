"""The research agent's slash commands.

Two kinds. *Conversational* commands (`/done`, `/waive`, `/rubric`,
`/approve`) change the interview, so the agent must see them: they are
forwarded to `handle()` as a turn (`ConsoleEffect.agent_turn`) — the same
path the eval ScriptedUser drives. *Operator* commands (`/cancel`,
`/history`, `/show`) read or act on the agent directly and never become a
turn.
"""

from collections.abc import Callable
from typing import Any

from agent_platform.interface.commands import CommandResult, ConsoleEffect
from agent_platform.interface.commands.types import CommandError

AgentRef = Callable[[], Any]


class ForwardedCommand:
    """A conversational act: validated here, handled by the agent's graph."""

    group = "Research"
    # The console shows it once, as the turn it becomes, not also as a command.
    forwards_turn = True

    def __init__(self, name: str, summary: str, usage: str, needs_args: bool = False) -> None:
        self.name = name
        self.summary = summary
        self.usage = usage
        self._needs_args = needs_args

    async def run(self, args: list[str]) -> CommandResult:
        if self._needs_args and not args:
            raise CommandError(self.usage)
        line = " ".join([f"/{self.name}", *args])
        return CommandResult(title=f"/{self.name}", effect=ConsoleEffect(agent_turn=line))


class CancelCommand:
    """Cancel the run in progress (also bound to a key while a run has the input)."""

    group = "Research"

    def __init__(self, name: str, summary: str, agent: AgentRef, replies: tuple[str, str]) -> None:
        self.name = name
        self.summary = summary
        self.usage = f"/{name}"
        self._agent = agent
        self._stopped, self._idle = replies

    async def run(self, args: list[str]) -> CommandResult:
        agent = self._agent()
        stopped = bool(agent is not None and agent.cancel())
        return CommandResult(title=f"/{self.name}").line(self._stopped if stopped else self._idle)
