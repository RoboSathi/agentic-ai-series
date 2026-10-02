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
from rich.text import Text

from agentic_commons.config import get_settings
from agentic_commons.config.settings import CheckStyleSettings
from agentic_commons.llm.benchmark import BenchmarkResult, benchmark
from agentic_commons.llm.local_probes import LocalProbeResult, build_local_probes
from agentic_commons.llm.probes import EndpointProbe, ProbeResult, build_probes
from agentic_commons.resilience.rate_limiter import get_rate_limiter

_PENDING = "..."
_PENDING_STATUS = "pending"
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
    styles = _styles()
    if failed:
        console.print(Text(f"✗ Failed: {', '.join(failed)}", style=styles.failure))
        return 1
    console.print(
        Text(
            "✓ Every required endpoint answered and every local model ran",
            style=styles.status["ok"],
        )
    )
    return 0


def _styles() -> CheckStyleSettings:
    return get_settings().llm.check.styles


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
    styles = _styles()
    if limiter is None:
        console.print(Text("Rate limiting is off", style=styles.status[_PENDING_STATUS]))
        return
    for name, bucket in settings.buckets.items():
        console.print(
            Text.assemble(
                ("Rate limit ", styles.detail),
                (name, styles.name),
                (
                    f": {bucket.max_requests} requests / {settings.window_seconds:.0f}s",
                    styles.model,
                ),
                (f", shared via {settings.resolved_store_path}", styles.target),
            )
        )


def _table(title: str, columns: tuple[str, ...]) -> Table:
    styles = _styles()
    table = Table(
        title=title,
        title_style=styles.title,
        border_style=styles.border,
        header_style=styles.header,
    )
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


def _status(status: str) -> Text:
    return Text(status, style=_styles().status.get(status, ""))


def _latency(ms: float, fast: float, slow: float, shown: str) -> Text:
    """A duration coloured by band: under `fast` fast, over `slow` slow, else medium."""
    styles = _styles()
    style = styles.fast if ms < fast else styles.slow if ms > slow else styles.medium
    return Text(shown, style=style)


def _ms(ms: float) -> Text:
    styles = _styles()
    return _latency(ms, styles.fast_ms, styles.slow_ms, f"{ms:.0f} ms")


def _seconds(seconds: float) -> Text:
    styles = _styles()
    return _latency(
        seconds, styles.bench_fast_seconds, styles.bench_slow_seconds, f"{seconds:.1f}s"
    )


def _endpoint_table(probes: list[EndpointProbe], arrived: dict[str, ProbeResult]) -> Table:
    styles = _styles()
    table = _table(
        "Hosted endpoints", ("probe", "endpoint", "model", "status", "required", "latency")
    )
    for probe in probes:
        result = arrived.get(probe.name)
        if result is None:
            pending = Text(_PENDING, style=styles.status[_PENDING_STATUS])
            table.add_row(Text(probe.name, style=styles.name), "", "", pending, "", "")
            continue
        table.add_row(
            Text(result.name, style=styles.name),
            Text(result.target, style=styles.target),
            Text(result.model, style=styles.model),
            _status(result.status),
            Text("yes", style=styles.required)
            if result.required
            else Text("no", style=styles.optional),
            _ms(result.latency_ms) if result.status != "skipped" else "",
        )
    return table


def _print_failures(console: Console, results: list[ProbeResult]) -> None:
    """Full error text below the table, where it can wrap without squeezing it."""
    styles = _styles()
    for result in results:
        if result.status == "failed":
            status = f" [HTTP {result.http_status}]" if result.http_status else ""
            console.print(
                Text.assemble(
                    (result.name, styles.failure),
                    (f" ({result.model})", styles.model),
                    (f"{status}: ", styles.failure),
                    (result.detail, styles.detail),
                )
            )


def _local_table(results: list[LocalProbeResult]) -> Table:
    styles = _styles()
    table = _table(
        "Local models (loaded and run once)",
        ("model", "name", "path", "status", "load", "inference", "output"),
    )
    for result in results:
        ran = result.status != "failed"
        table.add_row(
            Text(result.name, style=styles.name),
            Text(result.model, style=styles.model),
            Text(result.path, style=styles.target),
            _status(result.status),
            Text(f"{result.load_ms:.0f} ms", style=styles.detail) if ran else "",
            _ms(result.infer_ms) if ran else "",
            Text(result.detail, style=styles.detail if ran else styles.failure),
        )
    return table


def _bench_table(results: list[BenchmarkResult]) -> Table:
    styles = _styles()
    table = _table(
        "Chat model latency (judge-sized request, sequential calls)",
        ("model", "used as", "calls", "p50", "p95", "completion tok/s", "errors"),
    )
    for result in sorted(results, key=lambda item: item.p50_seconds):
        measured = result.samples - result.failures
        extras = f" {json.dumps(result.extra_body, sort_keys=True)}" if result.extra_body else ""
        calls_style = styles.status["ok"] if not result.failures else styles.failure
        table.add_row(
            Text(result.model + extras, style=styles.model),
            Text(", ".join(result.uses), style=styles.name),
            Text(f"{measured}/{result.samples}", style=calls_style),
            _seconds(result.p50_seconds) if measured else "",
            _seconds(result.p95_seconds) if measured else "",
            Text(f"{result.completion_tokens_per_second:.0f}", style=styles.detail)
            if measured
            else "",
            Text(result.detail, style=styles.failure),
        )
    return table


def main() -> int:
    """Synchronous wrapper so this stays a plain console entry point."""
    parser = argparse.ArgumentParser(description="Probe, run and optionally benchmark models.")
    parser.add_argument("--bench", action="store_true", help="time each chat model too")
    return asyncio.run(check(bench=parser.parse_args().bench))


if __name__ == "__main__":
    sys.exit(main())
