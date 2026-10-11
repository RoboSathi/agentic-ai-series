"""Commands that inspect the harness (LLD §20).

Each reads one subsystem and formats what it finds. They query directly
rather than going through the agent: an operator asking what is in the
audit ledger wants the ledger's answer, not a model's summary of it.
"""

from typing import Any, cast, get_args

from agent_platform.audit import (
    AuditEntry,
    AuditReader,
    Decision,
    DuckDbAuditLedger,
    SqliteAuditLedger,
)
from agent_platform.audit.view import AuditQuery, AuditTrail, table_lines, trace_tree_lines
from agent_platform.commands.types import CommandError, CommandResult
from agent_platform.config import get_settings
from agent_platform.config.yaml_config import config_sources, load_config
from agent_platform.gatekeeper.types import Standing
from agent_platform.observability.store import DuckDbObservabilityStore

_DECISIONS: tuple[str, ...] = get_args(Decision)
_STANDINGS: tuple[str, ...] = get_args(Standing)


class ConfigCommand:
    """Show configuration, whole or by section."""

    group = "Inspect"
    name = "config"
    summary = "Show configuration values"
    usage = "/config [section]   e.g. /config gatekeeper"

    async def run(self, args: list[str]) -> CommandResult:
        config = load_config()
        if not args:
            result = CommandResult(title="Configuration sections")
            result.headers = ["section", "keys"]
            for section, body in sorted(config.items()):
                size = len(body) if isinstance(body, dict) else 1
                result.row(section, str(size))
            sources = ", ".join(path.name for path in config_sources() if path.exists())
            result.line(f"merged from: {sources}")
            result.line("Use /config <section> to see one.")
            return result

        section = args[0]
        if section not in config:
            # `/config feature <name> off` is the shape people reach for
            # first. Naming the command that actually moves a feature
            # costs one line and saves a trip to /help.
            hint = (
                " Features are shown with '/config features' and changed with '/harness'."
                if section.casefold().startswith("feature")
                else ""
            )
            raise CommandError(f"No config section named '{section}'.{hint}")
        if len(args) > 1:
            # Silently ignoring the rest would let `/config features
            # gatekeeper off` print a config dump and read as success, so
            # the operator walks away believing they changed something.
            raise CommandError(
                f"/config only shows configuration. To change a feature, "
                f"use '/harness {' '.join(args[1:])}'."
            )
        result = CommandResult(title=f"config.{section}")
        result.headers = ["key", "value"]
        for key, value in _flatten(config[section]):
            result.row(key, value)
        return result


class AuditCommand:
    """Show recent decisions from the audit trail — every ledger the agent writes to.

    The live ledger is the shared DuckDB one; offline work (a dream, a review
    from a terminal) writes to the agent's own SQLite ledger. Both are read
    and merged by time (`audit/view.py`), the same view `make audit <agent>`
    prints. Within one process DuckDB shares the database instance, so
    reading alongside the running agent takes no second lock.
    """

    group = "Inspect"
    name = "audit"

    def __init__(self, ledger: AuditReader | None = None) -> None:
        self._ledger = ledger
        self._trail: AuditTrail | None = None
        self._words = get_settings().commands.audit
        self.summary = self._words.summary
        self.usage = self._words.usage

    @property
    def trail(self) -> AuditTrail:
        """The ledgers to read, opened on first use (only the one given, when one was)."""
        if self._trail is None:
            readers: list[AuditReader] = (
                [self._ledger]
                if self._ledger is not None
                else [DuckDbAuditLedger(DuckDbObservabilityStore()), SqliteAuditLedger()]
            )
            self._trail = AuditTrail(readers)
        return self._trail

    async def run(self, args: list[str]) -> CommandResult:
        if args and args[0] == self._words.trace_keyword:
            return self._trace(args[1:])
        words = [arg for arg in args if not arg.isdigit()]
        decision, standing = self._narrowing(words[0] if words else None)
        entries = self.trail.entries(
            AuditQuery(limit=_limit(args), decision=decision, standing=standing)
        )
        narrowed = decision or standing
        title = self._words.list_title.format(
            n=len(entries),
            narrowed=self._words.narrowed_by.format(filter=narrowed) if narrowed else "",
        )
        return self._table(CommandResult(title=title), entries).line(self._words.trace_hint)

    def _trace(self, args: list[str]) -> CommandResult:
        if not args:
            raise CommandError(self._words.needs_trace_id.format(usage=self.usage))
        entries = self.trail.trace(args[0])
        result = CommandResult(
            title=self._words.trace_title.format(trace_id=args[0], n=len(entries))
        )
        if not entries:
            return result.line(self._words.empty)
        result.lines = trace_tree_lines(args[0], entries)
        return result

    def _narrowing(self, value: str | None) -> tuple[Decision | None, Standing | None]:
        """A decision or a standing to filter by; anything else is a mistake, not a no-op."""
        if value is None:
            return None, None
        if value in _DECISIONS:
            return cast(Decision, value), None
        if value in _STANDINGS:
            return None, cast(Standing, value)
        raise CommandError(self._words.unknown_filter.format(value=value, usage=self.usage))

    def _table(self, result: CommandResult, entries: list[AuditEntry]) -> CommandResult:
        if not entries:
            return result.line(self._words.empty)
        for line in table_lines(entries):
            result.line(line)
        return result


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a config subtree into readable key/value pairs."""
    if not isinstance(value, dict):
        return [(prefix or "value", _render(value))]
    pairs: list[tuple[str, str]] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            pairs.extend(_flatten(child, path))
        else:
            pairs.append((path, _render(child)))
    return pairs


def _render(value: Any) -> str:
    """Render a config value as one cell.

    No truncation here. The view knows the terminal width and elides to
    it; a command that pre-truncated would either duplicate that rule or
    contradict it.
    """
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _limit(args: list[str]) -> int:
    """The numeric limit in a command's arguments, if given."""
    for arg in args:
        if arg.isdigit():
            return int(arg)
    return get_settings().commands.recent_window
