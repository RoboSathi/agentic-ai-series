"""Commands about the console itself (LLD §20)."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_commons.commands.types import CommandResult, ConsoleEffect
from agentic_commons.config import REPO_ROOT, get_settings
from agentic_commons.config.yaml_config import config_sources

if TYPE_CHECKING:
    from agentic_commons.commands.registry import CommandRegistry
    from agentic_commons.host import AgentHost


class HelpCommand:
    """List every command, or explain one."""

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
        result.headers = ["command", "what it does"]
        for command in self._registry.all_commands():
            result.row(f"/{command.name}", command.summary)
        result.line("Anything not starting with / is a message to the agent.")
        return result


class SessionCommand:
    """Show what this console is connected to.

    The provider, the database and the config layers are the platform's
    to report. Who is on the other end of the conversation is not: a
    verified customer is one agent's vocabulary, a tenant or a case
    number would be another's. So the running agent contributes its own
    rows and this command prints them, rather than importing one agent
    to ask it.
    """

    name = "session"
    summary = "Model, config sources and data in use"
    usage = "/session"

    def __init__(self, host: "AgentHost | None" = None, agent: Any = None) -> None:
        self._host = host
        self._agent = agent

    def bind(self, host: "AgentHost | None", agent: Any) -> None:
        """Attach the running host and agent, or detach on rebuild."""
        self._host = host
        self._agent = agent

    async def run(self, args: list[str]) -> CommandResult:
        settings = get_settings()
        provider = settings.llm.providers[settings.llm.provider]
        result = CommandResult(title="Session")
        result.headers = ["field", "value"]
        for field, value in self._agent_rows():
            result.row(field, value)
        result.row("provider", settings.llm.provider)
        result.row("model", provider.model)
        result.row("max tokens", str(settings.llm.max_tokens))
        result.row("database", str(settings.persistence.resolved_db_path.name))
        result.row("policy version", settings.audit.policy_version)
        # Shown as repo-relative paths. Every layer's file is called
        # config.yaml, so bare names would list the same word four times
        # and say nothing about which files are live — and two episodes'
        # overlays differ only in the directory above them.
        result.row(
            "config",
            ", ".join(_relative(path) for path in config_sources() if path.exists()),
        )
        logging_settings = get_settings().logging
        # Levels apply to the .log files only; the console shows one fixed view.
        result.row("log level", f"file {logging_settings.file_level}")
        return result

    def _agent_rows(self) -> list[tuple[str, str]]:
        """Whatever the running agent wants shown, or nothing."""
        if self._host is None:
            return []
        return list(self._host.session_rows(self._agent))


class LogsCommand:
    """Show where the logs are and how big they have grown."""

    name = "logs"
    summary = "Log files on disk"
    usage = "/logs"

    async def run(self, args: list[str]) -> CommandResult:
        settings = get_settings().logging
        directory = settings.resolved_dir
        result = CommandResult(title=f"Logs — {directory}")
        if not directory.exists():
            return result.line("No logs written yet.")
        files = sorted(directory.glob("*.log"))
        if not files:
            return result.line("No logs written yet.")
        result.headers = ["file", "size"]
        for path in files:
            result.row(path.name, f"{path.stat().st_size / 1024:,.1f} KB")
        result.line("")
        result.line(
            f"The file log captures {settings.file_level}; the console shows "
            f"{settings.console_level}. Full LLM payloads land in llm.log."
        )
        return result


class ClearCommand:
    """Clear the transcript.

    The console does the clearing — a command cannot reach the widget,
    and giving it one would make every command a UI component. This
    returns the request; the console honours it. It is a command rather
    than a special case in the input handler so that `/clear` appears in
    `/help` and dispatches like everything else.
    """

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
