"""The greeting: how to use the agent, in a few lines.

Behaviour first (ask a question; `/research` for a deeper run), then the commands, then which
world it is searching. Limits, budgets, providers and refusals are in the configuration and the
logs, not repeated here. In demo mode the last block lists exactly the topics the replayed
fixture world holds, so it cannot drift from the truth.
"""

from agentic_commons.config.settings import AppConfig
from research_agent.config.settings import ResearchAgentSettings

_QA, _RESEARCH = "qa", "research"


def build_greeting(settings: AppConfig, agent: ResearchAgentSettings, notes: list[str]) -> str:
    """The greeting for this mode; `notes` (a missing-key warning) are appended as given."""
    lines = agent.greeting
    demo = agent.model_backend == "fixture" and settings.web.mode == "replay"
    examples = _demo_examples(agent) if demo else _live_examples(agent)
    parts = [
        lines.intro,
        lines.research_line,
        lines.examples.format(qa=examples[_QA], research=examples[_RESEARCH]),
        "",
        lines.commands,
        *notes,
        "",
        *_mode_lines(agent, demo),
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
