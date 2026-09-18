"""Commands about the console itself (LLD §20)."""

from typing import TYPE_CHECKING

from agentic_commons.commands.types import CommandResult
from agentic_commons.config import get_settings
from agentic_commons.config.yaml_config import config_sources

if TYPE_CHECKING:
    from agentic_commons.commands.registry import CommandRegistry


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
    """Show what this console is connected to."""

    name = "session"
    summary = "Model, config sources and data in use"
    usage = "/session"

    async def run(self, args: list[str]) -> CommandResult:
        from customer_support_agent.core.identity import current_customer

        settings = get_settings()
        provider = settings.llm.providers[settings.llm.provider]
        customer = current_customer()
        result = CommandResult(title="Session")
        result.headers = ["field", "value"]
        result.row(
            "customer", f"{customer.name} (#{customer.customer_id})" if customer else "not verified"
        )
        result.row("provider", settings.llm.provider)
        result.row("model", provider.model)
        result.row("max tokens", str(settings.llm.max_tokens))
        result.row("database", str(settings.persistence.resolved_db_path.name))
        result.row("policy version", settings.audit.policy_version)
        result.row("config", ", ".join(p.name for p in config_sources() if p.exists()))
        return result


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
        return CommandResult(clears_transcript=True)
