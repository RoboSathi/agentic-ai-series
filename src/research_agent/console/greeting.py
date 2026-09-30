"""The greeting: what this agent can research right now, and what it will not do.

Assembled from configured lines and the running mode rather than written
once, so it cannot drift from the truth: in demo mode it lists exactly the
topics the replayed fixture world holds; live, it names the search
providers, the budgets and the month's remaining search quota.
"""

from agentic_commons.config.settings import AppConfig
from research_agent.config.settings import ResearchAgentSettings

_SECONDS_PER_MINUTE = 60
_QA, _RESEARCH = "qa", "research"


def build_greeting(settings: AppConfig, agent: ResearchAgentSettings, notes: list[str]) -> str:
    """The greeting for this mode; `notes` (quota, warnings) are appended as given."""
    lines = agent.greeting
    demo = agent.model_backend == "fixture" and settings.web.mode == "replay"
    examples = _demo_examples(agent) if demo else _live_examples(agent)
    parts = [*_mode_lines(agent, demo), "", lines.modes_title]
    parts += [
        lines.qa_line.format(example=examples[_QA]),
        lines.research_line.format(example=examples[_RESEARCH]),
        "",
        _limits(agent),
        lines.wont,
        *notes,
        lines.commands,
    ]
    return "\n".join(parts)


def _mode_lines(agent: ResearchAgentSettings, demo: bool) -> list[str]:
    if not demo:
        return [agent.greeting.live_mode]
    from research_agent.fixture_world.world import load_world

    world = load_world(agent.resolved_fixture_world_path)
    topics = [
        agent.greeting.topic_line.format(title=t.title, example=t.example) for t in world.topics
    ]
    return [agent.greeting.demo_mode, *topics]


def _demo_examples(agent: ResearchAgentSettings) -> dict[str, str]:
    from research_agent.fixture_world.world import load_world

    return load_world(agent.resolved_fixture_world_path).examples


def _live_examples(agent: ResearchAgentSettings) -> dict[str, str]:
    return {option.mode: option.example for option in agent.scoping.reframe_options}


def _limits(agent: ResearchAgentSettings) -> str:
    budget = agent.budget
    return agent.greeting.limits.format(
        qa_searches=agent.qa.budget.max_searches,
        searches=budget.max_searches,
        fetches=budget.max_fetches,
        minutes=round(budget.max_seconds / _SECONDS_PER_MINUTE),
    )
