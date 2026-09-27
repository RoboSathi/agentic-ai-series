"""Commands that inspect the harness (LLD §20).

Each reads one subsystem and formats what it finds. They query directly
rather than going through the agent: an operator asking what is in the
audit ledger wants the ledger's answer, not a model's summary of it.
"""

from typing import Any

from agentic_commons.audit import Decision, SqliteAuditLedger
from agentic_commons.commands._pool import PoolBackedCommand
from agentic_commons.commands.types import CommandError, CommandResult
from agentic_commons.config import get_settings
from agentic_commons.config.yaml_config import config_sources, load_config


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


class AuditCommand(PoolBackedCommand):
    """Show recent decisions from the audit ledger."""

    group = "Inspect"
    name = "audit"
    summary = "Recent decisions from the audit trail"
    usage = "/audit [allow|deny|escalate] [limit]"

    async def run(self, args: list[str]) -> CommandResult:
        ledger = SqliteAuditLedger(self.pool)
        verdicts: tuple[Decision, ...] = ("allow", "deny", "escalate")
        wanted = next((v for v in verdicts if args and args[0] == v), None)
        limit = _limit(args)

        entries = [
            entry
            for verdict in ((wanted,) if wanted else verdicts)
            for entry in ledger.by_decision(verdict)
        ]
        result = CommandResult(
            title=f"Audit trail — {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"
            + (f" ({wanted})" if wanted else "")
        )
        if not entries:
            return result.line("Nothing recorded yet.")
        result.headers = ["when", "actor", "action", "verdict", "why"]
        for entry in entries[-limit:]:
            style = {"deny": "red", "escalate": "yellow"}.get(entry.decision, "")
            result.row(
                entry.timestamp.strftime("%H:%M:%S"),
                entry.actor,
                entry.action or "—",
                entry.decision,
                entry.rationale[:52],
                style=style,
            )
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
