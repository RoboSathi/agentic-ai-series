"""Dispatching slash commands (LLD §20).

The registry owns what commands exist and how a typed line becomes one.
It is deliberately the only place that parses a slash: the TUI asks
"is this a command?" and gets a result or nothing, so a future console or
API dispatches identically without reimplementing the grammar.
"""

from agent_platform.commands.data import MemoryCommand
from agent_platform.commands.evals import EvalCommand
from agent_platform.commands.harness import HarnessCommand
from agent_platform.commands.inspection import (
    AuditCommand,
    ConfigCommand,
)
from agent_platform.commands.quarantine import DeadLetterCommand
from agent_platform.commands.system import (
    ClearCommand,
    HelpCommand,
)
from agent_platform.commands.types import Command, CommandError, CommandResult
from agent_platform.logging import get_logger

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

    def argument_completions(self, line: str) -> list[str]:
        """What the named command offers for the argument being typed.

        Only a command that *has* a closed set answers this. Most do not:
        an order id or a status is a value, and guessing one would be
        guessing at intent. A feature path is different — the set is
        fixed and the operator is choosing from it, which is the same
        kind of completion as the command word itself.
        """
        head, _, rest = line.strip().lstrip(PREFIX).partition(" ")
        command = self._commands.get(head.casefold())
        offered = getattr(command, "argument_completions", None)
        if offered is None:
            return []
        stem = rest.rsplit(" ", maxsplit=1)[-1].casefold()
        return [value for value in offered() if value.startswith(stem)]

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
    """Every command the *platform* ships with.

    Agent-owned commands are not listed here and never were meant to
    be: they arrive from the running agent's `AgentHost`, which is what
    lets this registry serve an agent the platform has never heard of.
    """
    return [
        HelpCommand(),
        ConfigCommand(),
        AuditCommand(),
        DeadLetterCommand(),
        MemoryCommand(),
        EvalCommand(),
        HarnessCommand(),
        ClearCommand(),
    ]
