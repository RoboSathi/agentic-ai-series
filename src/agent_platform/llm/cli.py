"""Command-line entry point for verifying models: `make model-check` (alias `llm-check`).

Checks every model the whole system uses, in one table. The platform's configuration is read,
then each agent's layer over it (`--layer name=overlay.yaml`, named by the Makefile — the
platform names no agent); every hosted model (chat models and their fallbacks, agent roles,
the judge, embeddings, the security guards and their fallbacks) and every local model (NLI,
reranker, intent classifier, embedders, summarizer, Presidio) is listed once, with every use
of it, then probed once. Hosted probes start together and each row fills in as its endpoint
answers; each local model is loaded and run once.

Exits non-zero when a model something calls *first* fails (a primary chat model, a role, the
judge, an enabled guard, a local model). A failing fallback is reported, not fatal. With
`--bench`, each chat model is also timed on a judge-sized request.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from agent_platform.config import AppConfig, get_settings, reload_settings
from agent_platform.config.settings import CheckStyleSettings
from agent_platform.config.yaml_config import clear_overlays, use_overlay
from agent_platform.llm.benchmark import BenchmarkResult, ChatModel, benchmark
from agent_platform.llm.inventory import InventoryEntry, ModelUse, grouped, merge, uses_in
from agent_platform.llm.local_probes import LocalProbeResult, build_local_probes
from agent_platform.llm.probes import (
    ChatProbe,
    EmbeddingProbe,
    EndpointProbe,
    ProbeResult,
    guard_probe,
)
from agent_platform.resilience.rate_limiter import get_rate_limiter

_PENDING = "..."
_PENDING_STATUS = "pending"
_REFRESH_PER_SECOND = 8


Layer = tuple[str, str | None]


async def check(layers: Sequence[Layer] = (), bench: bool = False) -> int:
    """Inventory every layer's models, probe each once, report, optionally benchmark."""
    console = Console()
    uses, local = await _collect(layers)
    entries = merge(uses)
    results = await _probe_all(console, entries)
    _print_failures(console, results)
    console.print(_local_table(local))
    if bench:
        targets = [
            ChatModel(provider=e.provider, model=e.model, uses=e.described())
            for e in entries
            if e.kind == "chat"
        ]
        console.print(_bench_table(await benchmark(models=targets)))
    _print_budget(console)
    failed = [result.model for result in results if result.status == "failed" and result.required]
    failed += [f"local:{row.result.name}" for row in local if row.result.status == "failed"]
    styles = _styles()
    if failed:
        console.print(Text(f"✗ Failed: {', '.join(failed)}", style=styles.failure))
        return 1
    console.print(
        Text(
            "✓ Every model something depends on answered, and every local model ran",
            style=styles.status["ok"],
        )
    )
    return 0


class _LocalRow:
    """One local model's result and every use of it."""

    def __init__(self, result: LocalProbeResult) -> None:
        self.result = result
        # (use, layer) per configuration layer that uses this model.
        self.uses: list[tuple[str, str]] = []


async def _collect(layers: Sequence[Layer]) -> tuple[list[ModelUse], list[_LocalRow]]:
    """Each layer in turn: its hosted model uses, and its local models (run once each)."""
    platform = get_settings().llm.check.platform_layer
    uses: list[ModelUse] = []
    local: dict[tuple[str, str], _LocalRow] = {}
    try:
        for name, overlay in [(platform, None), *layers]:
            config = _apply(overlay)
            uses += uses_in(config, name)
            for probe in build_local_probes(config):
                key = (probe.name, probe.model)
                if key not in local:
                    local[key] = _LocalRow(await probe.probe())
                local[key].uses.append((config.llm.check.uses.get(probe.name, probe.name), name))
    finally:
        _apply(None)
    return uses, list(local.values())


def _apply(overlay: str | None) -> AppConfig:
    """Make `overlay` (or none) the configuration's only layer over the platform's."""
    clear_overlays()
    if overlay:
        use_overlay(overlay)
    return reload_settings()


def _probe_for(entry: InventoryEntry, config: AppConfig) -> EndpointProbe:
    name = "; ".join(entry.described())
    if entry.kind == "embedding":
        return EmbeddingProbe(config)
    if entry.kind == "guard":
        return guard_probe(config, entry.model, entry.required)
    return ChatProbe(name, entry.provider, entry.model, entry.required, config)


def _styles() -> CheckStyleSettings:
    return get_settings().llm.check.styles


async def _probe_all(console: Console, entries: list[InventoryEntry]) -> list[ProbeResult]:
    """Probe every model together; redraw the table as each answers. Rows keep inventory order."""
    config = get_settings()
    probes = [_probe_for(entry, config) for entry in entries]
    arrived: dict[int, ProbeResult] = {}

    async def run(index: int, probe: EndpointProbe) -> None:
        arrived[index] = await probe.probe()

    with Live(
        _endpoint_table(entries, arrived), console=console, refresh_per_second=_REFRESH_PER_SECOND
    ) as live:
        pending = [asyncio.create_task(run(i, probe)) for i, probe in enumerate(probes)]
        for finished in asyncio.as_completed(pending):
            await finished
            live.update(_endpoint_table(entries, arrived))
    return [arrived[i] for i in range(len(probes))]


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


def _endpoint_table(entries: list[InventoryEntry], arrived: dict[int, ProbeResult]) -> Table:
    styles = _styles()
    table = _table(
        "Hosted models (every use across the system)",
        ("model", "kind", "used for", "status", "required", "latency"),
    )
    for index, entry in enumerate(entries):
        result = arrived.get(index)
        status = (
            _status(result.status)
            if result is not None
            else Text(_PENDING, style=styles.status[_PENDING_STATUS])
        )
        table.add_row(
            Text(entry.model, style=styles.model),
            Text(entry.kind, style=styles.detail),
            Text("\n".join(entry.described()), style=styles.name),
            status,
            Text("yes", style=styles.required)
            if entry.required
            else Text("fallback", style=styles.optional),
            _ms(result.latency_ms) if result is not None and result.status != "skipped" else "",
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


def _local_table(rows: list[_LocalRow]) -> Table:
    styles = _styles()
    table = _table(
        "Local models (loaded and run once)",
        ("model", "used for", "path", "status", "load", "inference", "output"),
    )
    for row in rows:
        result = row.result
        ran = result.status != "failed"
        table.add_row(
            Text(result.model, style=styles.model),
            Text("\n".join(grouped(row.uses)), style=styles.name),
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


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous wrapper so this stays a plain console entry point."""
    parser = argparse.ArgumentParser(description="Probe, run and optionally benchmark models.")
    parser.add_argument("--bench", action="store_true", help="time each chat model too")
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        metavar="NAME=OVERLAY",
        help="an agent's config layer to include (repeatable)",
    )
    args = parser.parse_args(argv)
    layers = [_layer(spec) for spec in args.layer]
    return asyncio.run(check(layers, bench=args.bench))


def _layer(spec: str) -> Layer:
    name, _, overlay = spec.partition("=")
    return name, overlay or None


if __name__ == "__main__":
    sys.exit(main())
