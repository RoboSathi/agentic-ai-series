"""The phase-2 scaffold: the shell can resolve, build and talk to the research agent."""

from agentic_commons.config import get_settings
from agentic_commons.host import AgentHost, resolve_host
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter


async def test_the_overlay_names_the_research_host(ra_config: None) -> None:
    host = resolve_host(get_settings().agent.host)
    assert isinstance(host, AgentHost)
    assert get_settings().agent.id == "research_agent"
    assert host.title() and host.placeholder()
    host.prepare()
    assert host.commands() == [] and dict(host.stage_labels()) == {}
    feed = InMemoryLiveStatusFeed()
    agent = host.build(StageReporter(feed, "t"))
    reply = await agent.handle("hello", session_id="s")
    assert reply.response
    assert [event.stage for event in feed.events] == ["intake"]
