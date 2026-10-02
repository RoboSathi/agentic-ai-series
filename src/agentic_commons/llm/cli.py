"""Command-line entry point for verifying models: `make model-check` (alias `llm-check`).

Probes every hosted model the configuration names — the platform's chat
model, the eval judge, each model the running agent's roles use,
embeddings and the security guards — then loads and runs each local
model once. Every hosted probe starts at once and its row fills in the
moment its endpoint answers, so the run takes as long as the slowest
endpoint (bounded by `llm.check.timeout_seconds`), not the sum of them.
With `--bench`, each chat model is also timed on a judge-sized request.
Exits non-zero when a *required* endpoint fails, or a configured local
model fails to load.
"""

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.live import Live
from rich.table import Table

from agentic_commons.config import get_settings
from agentic_commons.llm.benchmark import BenchmarkResult, benchmark
from agentic_commons.llm.local_probes import LocalProbeResult, build_local_probes
from agentic_commons.llm.probes import EndpointProbe, ProbeResult, build_probes
from agentic_commons.resilience.rate_limiter import get_rate_limiter

_STATUS_STYLE = {"ok": "green", "failed": "red", "skipped": "dim", "downloaded": "cyan"}
_PENDING = "..."
_REFRESH_PER_SECOND = 8


async def check(bench: bool = False) -> int:
    """Probe every endpoint at once, run each local model, optionally benchmark, and report."""
    console = Console()
    probes = build_probes()
    results = await _probe_all(console, probes)
    _print_failures(console, results)
    local = [await probe.probe() for probe in build_local_probes()]
    console.print(_local_table(local))
    if bench:
        console.print(_bench_table(await benchmark()))
    _print_budget(console)
    failed = [result.name for result in results if result.status == "failed" and result.required]
    failed += [f"local:{result.name}" for result in local if result.status == "failed"]
    if failed:
        console.print(f"[red]✗ Failed:[/red] {', '.join(failed)}")
        return 1
    console.print("[green]✓ Every required endpoint answered and every local model ran[/green]")
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


def _local_table(results: list[LocalProbeResult]) -> Table:
    table = Table(title="Local models (loaded and run once)")
    for column in ("model", "name", "path", "status", "load", "inference", "output"):
        table.add_column(column, overflow="fold")
    for result in results:
        style = _STATUS_STYLE[result.status]
        ran = result.status != "failed"
        table.add_row(
            result.name,
            result.model,
            result.path,
            f"[{style}]{result.status}[/{style}]",
            f"{result.load_ms:.0f} ms" if ran else "",
            f"{result.infer_ms:.0f} ms" if ran else "",
            result.detail,
        )
    return table


def _bench_table(results: list[BenchmarkResult]) -> Table:
    table = Table(title="Chat model latency (judge-sized request, sequential calls)")
    for column in ("model", "used as", "calls", "p50", "p95", "completion tok/s", "errors"):
        table.add_column(column, overflow="fold")
    for result in sorted(results, key=lambda item: item.p50_seconds):
        measured = result.samples - result.failures
        extras = f" {json.dumps(result.extra_body, sort_keys=True)}" if result.extra_body else ""
        table.add_row(
            result.model + extras,
            ", ".join(result.uses),
            f"{measured}/{result.samples}",
            f"{result.p50_seconds:.1f}s" if measured else "",
            f"{result.p95_seconds:.1f}s" if measured else "",
            f"{result.completion_tokens_per_second:.0f}" if measured else "",
            result.detail,
        )
    return table


def main() -> int:
    """Synchronous wrapper so this stays a plain console entry point."""
    parser = argparse.ArgumentParser(description="Probe, run and optionally benchmark models.")
    parser.add_argument("--bench", action="store_true", help="time each chat model too")
    return asyncio.run(check(bench=parser.parse_args().bench))


if __name__ == "__main__":
    sys.exit(main())
