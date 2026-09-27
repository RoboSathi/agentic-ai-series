"""Command-line entry point for verifying hosted endpoints: `make llm-check`.

Probes every hosted model the configuration names — the agent's chat
model, the eval judge, embeddings and the security guards — and lists
the local models the harness loads from disk. Exits non-zero when a
*required* endpoint fails.
"""

import asyncio
import sys

from rich.console import Console
from rich.table import Table

from agentic_commons.config import get_settings
from agentic_commons.llm.probes import ProbeResult, build_probes, local_models
from agentic_commons.resilience.rate_limiter import get_rate_limiter

_STATUS_STYLE = {"ok": "green", "failed": "red", "skipped": "dim"}


async def check() -> int:
    """Probe every endpoint (sequentially — they share one budget) and report."""
    console = Console()
    _print_budget(console)
    results = [await probe.probe() for probe in build_probes()]
    console.print(_endpoint_table(results))
    _print_failures(console, results)
    console.print(_local_table())
    failed = [result.name for result in results if result.status == "failed" and result.required]
    if failed:
        console.print(f"[red]✗ Required endpoints failed:[/red] {', '.join(failed)}")
        return 1
    console.print("[green]✓ Every required endpoint answered[/green]")
    return 0


def _print_budget(console: Console) -> None:
    limiter = get_rate_limiter()
    settings = get_settings().resilience.rate_limit
    if limiter is None:
        console.print("[yellow]Rate limiting is off[/yellow]")
        return
    for name, bucket in settings.buckets.items():
        console.print(
            f"Rate limit [bold]{name}[/bold]: {bucket.max_requests} requests / "
            f"{settings.window_seconds:.0f}s, shared via {settings.resolved_store_path}"
        )


def _endpoint_table(results: list[ProbeResult]) -> Table:
    table = Table(title="Hosted endpoints")
    for column in ("probe", "endpoint", "model", "status", "required", "http", "latency"):
        table.add_column(column, overflow="fold")
    for result in results:
        style = _STATUS_STYLE[result.status]
        table.add_row(
            result.name,
            result.target,
            result.model,
            f"[{style}]{result.status}[/{style}]",
            "yes" if result.required else "no",
            str(result.http_status or ""),
            f"{result.latency_ms:.0f} ms" if result.status != "skipped" else "",
        )
    return table


def _print_failures(console: Console, results: list[ProbeResult]) -> None:
    """Full error text below the table, where it can wrap without squeezing it."""
    for result in results:
        if result.status == "failed":
            console.print(f"[red]{result.name}[/red] ({result.model}): {result.detail}")


def _local_table() -> Table:
    table = Table(title="Local models")
    for column in ("model", "path", "present"):
        table.add_column(column)
    for model in local_models():
        mark = "[green]yes[/green]" if model.present else "[red]missing[/red]"
        table.add_row(model.name, model.path, mark)
    return table


def main() -> int:
    """Synchronous wrapper so this stays a plain console entry point."""
    return asyncio.run(check())


if __name__ == "__main__":
    sys.exit(main())
