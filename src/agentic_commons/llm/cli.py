"""Command-line entry point for verifying hosted endpoints: `make llm-check` (alias `model-check`).

Probes every hosted model the configuration names — the agent's chat
model, the eval judge, embeddings and the security guards — and lists
the local models the harness loads from disk. Every probe starts at once
and its row fills in the moment its endpoint answers, so the run takes as
long as the slowest endpoint (bounded by `llm.check.timeout_seconds`), not
the sum of them. Exits non-zero when a *required* endpoint fails.
"""

import asyncio
import sys

from rich.console import Console
from rich.live import Live
from rich.table import Table

from agentic_commons.config import get_settings
from agentic_commons.llm.probes import EndpointProbe, ProbeResult, build_probes, local_models
from agentic_commons.resilience.rate_limiter import get_rate_limiter

_STATUS_STYLE = {"ok": "green", "failed": "red", "skipped": "dim"}
_PENDING = "..."
_REFRESH_PER_SECOND = 8


async def check() -> int:
    """Probe every endpoint at once, showing each result as it arrives, and report."""
    console = Console()
    probes = build_probes()
    results = await _probe_all(console, probes)
    _print_failures(console, results)
    console.print(_local_table())
    _print_budget(console)
    failed = [result.name for result in results if result.status == "failed" and result.required]
    if failed:
        console.print(f"[red]✗ Required endpoints failed:[/red] {', '.join(failed)}")
        return 1
    console.print("[green]✓ Every required endpoint answered[/green]")
    return 0


async def _probe_all(console: Console, probes: list[EndpointProbe]) -> list[ProbeResult]:
    """Run the probes together; redraw the table whenever one finishes. Rows keep config order."""
    arrived: dict[str, ProbeResult] = {}

    async def run(probe: EndpointProbe) -> None:
        arrived[probe.name] = await probe.probe()

    with Live(
        _endpoint_table(probes, arrived), console=console, refresh_per_second=_REFRESH_PER_SECOND
    ) as live:
        pending = [asyncio.create_task(run(probe)) for probe in probes]
        for finished in asyncio.as_completed(pending):
            await finished
            live.update(_endpoint_table(probes, arrived))
    return [arrived[probe.name] for probe in probes]


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


def _endpoint_table(probes: list[EndpointProbe], arrived: dict[str, ProbeResult]) -> Table:
    table = Table(title="Hosted endpoints")
    for column in ("probe", "endpoint", "model", "status", "required", "latency"):
        table.add_column(column, overflow="fold")
    for probe in probes:
        result = arrived.get(probe.name)
        if result is None:
            table.add_row(probe.name, "", "", f"[yellow]{_PENDING}[/yellow]", "", "")
            continue
        style = _STATUS_STYLE[result.status]
        table.add_row(
            result.name,
            result.target,
            result.model,
            f"[{style}]{result.status}[/{style}]",
            "yes" if result.required else "no",
            f"{result.latency_ms:.0f} ms" if result.status != "skipped" else "",
        )
    return table


def _print_failures(console: Console, results: list[ProbeResult]) -> None:
    """Full error text below the table, where it can wrap without squeezing it."""
    for result in results:
        if result.status == "failed":
            status = f" [HTTP {result.http_status}]" if result.http_status else ""
            console.print(f"[red]{result.name}[/red] ({result.model}){status}: {result.detail}")


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
