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
from agentic_commons.observability import LittlesLawQueueModel, SLORegistry


class ConfigCommand:
    """Show configuration, whole or by section."""

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
            raise CommandError(f"No config section named '{section}'")
        result = CommandResult(title=f"config.{section}")
        result.headers = ["key", "value"]
        for key, value in _flatten(config[section]):
            result.row(key, value)
        return result


class AuditCommand(PoolBackedCommand):
    """Show recent decisions from the audit ledger."""

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


class SloCommand:
    """Show SLO targets and how each is standing."""

    name = "slo"
    summary = "SLO targets and error budgets"
    usage = "/slo"

    async def run(self, args: list[str]) -> CommandResult:
        registry = SLORegistry()
        targets = get_settings().observability.slo_targets
        result = CommandResult(title=f"{len(targets)} SLOs")
        result.headers = ["metric", "target", "observed", "budget", "state"]
        for status in registry.all_statuses():
            observed = f"{status.observed:.3f}" if status.sample_count else "—"
            result.row(
                status.metric,
                f"{status.target:.2f}",
                observed,
                f"{status.error_budget_remaining:.0%}",
                "ok" if status.within_target else "BREACH",
                style="" if status.within_target else "red",
            )
        result.line("Targets of 0.00 are zero-tolerance: any occurrence is a breach.")
        return result


class QueueCommand(PoolBackedCommand):
    """Show the human review queue and the staffing it implies."""

    name = "queue"
    summary = "Human review queue and its staffing maths"
    usage = "/queue [arrivals_per_hour] [minutes_each]"

    async def run(self, args: list[str]) -> CommandResult:
        from customer_support_agent.core.repositories import HitlQueueRepository

        pending = HitlQueueRepository(self.pool).pending()
        result = CommandResult(title=f"Review queue — {len(pending)} pending")
        if pending:
            result.headers = ["id", "order", "customer", "amount", "raised"]
            for request in pending[: get_settings().commands.recent_window]:
                result.row(
                    str(request.id),
                    str(request.order_id),
                    str(request.user_id),
                    f"${request.requested_amount:,.2f}",
                    request.created_at[:16],
                )
        else:
            result.line("Nothing awaiting review.")

        arrivals = float(args[0]) if args else max(len(pending), 1) * 1.0
        minutes = float(args[1]) if len(args) > 1 else 4.0
        capacity = LittlesLawQueueModel().capacity_for(arrivals, minutes)
        result.line("")
        result.line(
            f"Little's Law at {arrivals:g}/hr × {minutes:g} min: "
            f"{capacity.reviewer_hours_per_hour:.2f} reviewer-hours per hour "
            f"({capacity.reviewers_required} people)"
        )
        return result


class CostCommand:
    """Show what the configured model costs to run."""

    name = "cost"
    summary = "Model pricing and the cost-per-success denominator"
    usage = "/cost"

    async def run(self, args: list[str]) -> CommandResult:
        settings = get_settings().llm
        result = CommandResult(title="Cost")
        result.headers = ["provider", "model", "input /1k", "output /1k"]
        for name, provider in settings.providers.items():
            marker = " (active)" if name == settings.provider else ""
            result.row(
                f"{name}{marker}",
                provider.model,
                f"${provider.input_cost_per_1k_tokens:.4f}",
                f"${provider.output_cost_per_1k_tokens:.4f}",
            )
        result.line("")
        result.line("Cost is tracked per request and divided by *successful* outcomes,")
        result.line("never by call count — a cheaper wrong answer is not a saving.")
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
