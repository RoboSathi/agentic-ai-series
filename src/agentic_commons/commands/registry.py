"""Dispatching slash commands (LLD §20).

The registry owns what commands exist and how a typed line becomes one.
It is deliberately the only place that parses a slash: the TUI asks
"is this a command?" and gets a result or nothing, so a future console or
API dispatches identically without reimplementing the grammar.
"""

from agentic_commons.commands.data import MemoryCommand, OrdersCommand, UsersCommand
from agentic_commons.commands.harness import (
    CascadeCommand,
    EvalCommand,
    GatekeeperCommand,
    PatternsCommand,
    ToolsCommand,
)
from agentic_commons.commands.inspection import (
    AuditCommand,
    ConfigCommand,
    CostCommand,
    QueueCommand,
    SloCommand,
)
from agentic_commons.commands.system import (
    ClearCommand,
    HelpCommand,
    LogsCommand,
    SessionCommand,
)
from agentic_commons.commands.types import Command, CommandError, CommandResult
from agentic_commons.logging import get_logger

PREFIX = "/"


class CommandRegistry:
    """Every console command, and the dispatch for one typed line."""

    def __init__(self, commands: list[Command] | None = None) -> None:
        self._commands: dict[str, Command] = {}
        for command in commands or _default_commands():
            self.register(command)
        # Help needs the registry to list it, so it is wired afterwards.
        help_command = self._commands.get("help")
        if isinstance(help_command, HelpCommand):
            help_command.bind(self)
        self._logger = get_logger("tui")

    def register(self, command: Command) -> None:
        """Add a command under its name."""
        self._commands[command.name] = command

    def all_commands(self) -> list[Command]:
        """Every command, alphabetically."""
        return [self._commands[name] for name in sorted(self._commands)]

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    @staticmethod
    def is_command(line: str) -> bool:
        """Whether a typed line is a command rather than a message."""
        return line.strip().startswith(PREFIX)

    def completions(self, partial: str) -> list[str]:
        """Command names matching a partially typed line."""
        stem = partial.strip().lstrip(PREFIX).casefold()
        return [name for name in sorted(self._commands) if name.startswith(stem)]

    async def dispatch(self, line: str) -> CommandResult:
        """Run the command a line names.

        An unknown command suggests near matches rather than failing
        blankly — a typo should cost a glance, not a trip to the docs.
        """
        parts = line.strip().lstrip(PREFIX).split()
        if not parts:
            return await self.dispatch("/help")
        name, args = parts[0].casefold(), parts[1:]
        command = self._commands.get(name)
        if command is None:
            suggestions = self.completions(name)
            result = CommandResult(title=f"Unknown command '/{name}'", ok=False)
            if suggestions:
                result.line(f"Did you mean: {', '.join('/' + s for s in suggestions)}")
            return result.line("Type /help to see everything.")
        try:
            return await command.run(args)
        except CommandError as exc:
            return CommandResult(title=f"/{name}", ok=False).line(str(exc)).line(command.usage)
        except Exception as exc:
            self._logger.exception("Command failed", command=name)
            return CommandResult(title=f"/{name} failed", ok=False).line(str(exc))


def _default_commands() -> list[Command]:
    """Every command the console ships with."""
    return [
        HelpCommand(),
        ConfigCommand(),
        AuditCommand(),
        SloCommand(),
        QueueCommand(),
        CostCommand(),
        UsersCommand(),
        OrdersCommand(),
        MemoryCommand(),
        EvalCommand(),
        CascadeCommand(),
        ToolsCommand(),
        GatekeeperCommand(),
        PatternsCommand(),
        SessionCommand(),
        LogsCommand(),
        ClearCommand(),
    ]
