"""Commands about the console itself (LLD §20)."""

from pathlib import Path
from typing import TYPE_CHECKING

from agent_platform.config import REPO_ROOT
from agent_platform.interface.commands.types import CommandResult, ConsoleEffect

if TYPE_CHECKING:
    from agent_platform.interface.commands.registry import CommandRegistry


# Where a command that declares no group is listed.
_DEFAULT_GROUP = "Agent"


class HelpCommand:
    """List every command, or explain one."""

    group = "Console"
    name = "help"
    summary = "List commands, or explain one"
    usage = "/help [command]"

    def __init__(self) -> None:
        self._registry: "CommandRegistry | None" = None

    def bind(self, registry: "CommandRegistry") -> None:
        """Give help the registry it describes."""
        self._registry = registry

    async def run(self, args: list[str]) -> CommandResult:
        if self._registry is None:
            return CommandResult(title="Commands", ok=False).line("No registry bound.")
        if args:
            command = self._registry.get(args[0].lstrip("/").casefold())
            if command is None:
                return CommandResult(title=f"No command '/{args[0]}'", ok=False)
            return (
                CommandResult(title=f"/{command.name}")
                .line(command.summary)
                .line("")
                .line(command.usage)
            )
        result = CommandResult(title="Commands")
        result.headers = ["group", "command", "what it does"]
        # One line per registered command, grouped; the registry is the
        # only list of commands there is.
        commands = sorted(
            self._registry.all_commands(),
            key=lambda command: (getattr(command, "group", _DEFAULT_GROUP), command.name),
        )
        for command in commands:
            result.row(
                getattr(command, "group", _DEFAULT_GROUP), f"/{command.name}", command.summary
            )
        result.line("Anything not starting with / is a message to the agent.")
        return result


class ClearCommand:
    """Clear the transcript.

    The console does the clearing — a command cannot reach the widget,
    and giving it one would make every command a UI component. This
    returns the request; the console honours it. It is a command rather
    than a special case in the input handler so that `/clear` appears in
    `/help` and dispatches like everything else.
    """

    group = "Console"
    name = "clear"
    summary = "Clear the transcript"
    usage = "/clear"

    async def run(self, args: list[str]) -> CommandResult:
        return CommandResult(effect=ConsoleEffect(clear_transcript=True))


def _relative(path: Path) -> str:
    """One config file, named from the repo root so it is unambiguous."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        # An overlay from outside the repo — a machine-local path, say.
        # Its full location is the only honest thing to show.
        return str(path)
