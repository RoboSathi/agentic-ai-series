"""`make data ra`: record the fixture world, then prove it replays.

Recording runs the platform's real web stack in `record` mode — the Tavily
adapter, the HTTP fetcher with its reader fallback, the fixture store —
against `MockWeb`, an in-process stand-in for the internet that serves
`world.yaml`. So the fixtures are exactly what the adapters produce, and
recording them is itself a test of the adapters.

Validation then reopens the set in `replay` mode and checks every hazard
the evals rely on is really there.
"""

import asyncio
import shutil
import sys
from pathlib import Path

import httpx
from rich.console import Console

from agentic_commons.config import get_settings
from agentic_commons.web import (
    FallbackFetcher,
    FixtureStore,
    HttpFetcher,
    ReaderFetcher,
    ReplayMissError,
    SearchRequest,
    TavilyProvider,
    WebGateway,
)
from research_agent.config import get_agent_settings
from research_agent.fixture_world.mock_web import MockWeb
from research_agent.fixture_world.world import World, load_world

_RECORDER_KEY = "fixture-recorder"
console = Console()


async def record(world: World, root: Path, max_results: int) -> FixtureStore:
    """Write the fixture set by recording every query and page through the real adapters."""
    settings = get_settings().web
    fixtures = FixtureStore(root, world.set_name)
    shutil.rmtree(fixtures.folder, ignore_errors=True)
    mock = MockWeb(world, f"{settings.tavily.base_url.rstrip('/')}{settings.tavily.endpoint}")
    fetch = settings.fetch.model_copy(
        update={"reader_url_template": world.reader_url_template, "reader_enabled": True}
    )
    async with httpx.AsyncClient(transport=mock.transport()) as client:
        gateway = WebGateway(
            "record",
            fixtures=fixtures,
            provider=TavilyProvider(settings.tavily, client, api_key=_RECORDER_KEY),
            fetcher=FallbackFetcher(
                HttpFetcher(fetch, client), ReaderFetcher(fetch, client), fetch
            ),
        )
        for query in world.queries:
            await gateway.search(SearchRequest(query=query.query, max_results=max_results))
        for page in world.pages:
            await gateway.fetch(page.url)
    fixtures.write_manifest(world.version, world.description)
    return fixtures


async def validate(world: World, fixtures: FixtureStore, max_results: int) -> list[str]:
    """Every hazard the evals depend on, checked against the replayed set."""
    replay = WebGateway("replay", fixtures=fixtures)
    problems = [
        f"no replay for query '{q.query}'"
        for q in world.queries
        if not await _replays(replay, q.query, max_results)
    ]
    for page in world.pages:
        replayed = await replay.fetch(page.url)
        if page.status >= 400 and replayed.status != page.status:
            problems.append(f"{page.url}: expected HTTP {page.status}")
        if page.js_only and replayed.via != "reader":
            problems.append(f"{page.url}: JS page not read through the reader")
    return problems + _hostile(world) + _summaries(world) + await _miss_is_loud(replay, max_results)


async def _replays(gateway: WebGateway, query: str, max_results: int) -> bool:
    try:
        await gateway.search(SearchRequest(query=query, max_results=max_results))
    except ReplayMissError:
        return False
    return True


def _hostile(world: World) -> list[str]:
    kinds = {page.hostile for page in world.hostile_pages()}
    wanted = {"instruction_override", "exfiltration", "fake_system_notice"}
    return [f"missing hostile page: {kind}" for kind in sorted(wanted - kinds)]


def _summaries(world: World) -> list[str]:
    bodies = " ".join(page.body for page in world.pages)
    return [
        f"provider answer appears on a page: {q.answer}"
        for q in world.queries
        if q.answer and q.answer in bodies
    ]


async def _miss_is_loud(gateway: WebGateway, max_results: int) -> list[str]:
    try:
        probe = get_agent_settings().evals.probe_query
        await gateway.search(SearchRequest(query=probe, max_results=max_results))
    except ReplayMissError:
        return []
    return ["a replay miss did not raise"]


def ensure_recorded() -> FixtureStore:
    """The configured fixture set, recorded first if it has never been (idempotent)."""
    web = get_settings().web
    fixtures = FixtureStore(web.resolved_fixtures_dir, web.fixture_set)
    if fixtures.manifest() is None:
        world = load_world(get_agent_settings().resolved_fixture_world_path)
        max_results = get_agent_settings().research.max_results_per_search
        fixtures = asyncio.run(record(world, web.resolved_fixtures_dir, max_results))
    return fixtures


def main() -> int:
    world = load_world(get_agent_settings().resolved_fixture_world_path)
    max_results = get_agent_settings().research.max_results_per_search
    root = get_settings().web.resolved_fixtures_dir
    fixtures = asyncio.run(record(world, root, max_results))
    problems = asyncio.run(validate(world, fixtures, max_results))
    searches, fetches = len(fixtures.keys("search")), len(fixtures.keys("fetch"))
    console.print(
        f"[bold]{world.set_name}[/bold] v{world.version}: {len(world.pages)} pages, "
        f"{searches} searches and {fetches} fetches recorded under {fixtures.folder}"
    )
    for problem in problems:
        console.print(f"[red]✗ {problem}[/red]")
    if not problems:
        console.print(
            "[green]✓ fixture world valid: replay complete, hazards present, misses loud[/green]"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
