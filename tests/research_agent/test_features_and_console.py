"""Feature toggles (withheld = never built), the injection ablations, cancel, and the console host."""

import pytest

from agentic_commons.commands import CommandResult
from agentic_commons.commands.types import CommandError
from agentic_commons.host import CancellableHost, GreetingHost, StatusHost
from research_agent.config import get_agent_settings
from research_agent.console import ResearchHost


def _run_nodes(h: object) -> set[str]:
    return set(h.agent._runs._graph.get_graph().nodes)  # type: ignore[attr-defined]


def _scope_nodes(h: object) -> set[str]:
    return set(h.agent._scoping._graph.get_graph().nodes)  # type: ignore[attr-defined]


class TestConsoleHost:
    def test_the_host_offers_every_hook(self, ra_config: None) -> None:
        host = ResearchHost()
        assert (
            isinstance(host, GreetingHost)
            and isinstance(host, StatusHost)
            and isinstance(host, CancellableHost)
        )
        assert host.greeting() and host.cancel_key() == get_agent_settings().cancel.key
        assert set(get_agent_settings().tui.stage_labels) <= set(host.stage_labels())
        assert host.status_lines("s") == ([], []) and host.cancel() is False

    async def test_conversational_commands_are_forwarded_as_turns(self, ra_config: None) -> None:
        commands = {c.name: c for c in ResearchHost().commands()}
        done: CommandResult = await commands["done"].run([])
        assert done.effect.agent_turn == "/done"
        edit = await commands["rubric"].run(["add", "lead", "times"])
        assert edit.effect.agent_turn == "/rubric add lead times"
        with pytest.raises(CommandError):
            await commands["waive"].run([])
        idle = await commands["cancel"].run([])
        assert idle.lines == [get_agent_settings().messages.no_active_run]


def _any_contract() -> object:
    from research_agent.domain.rubric import Contract, Rubric
    from research_agent.domain.scope import ScopeBrief

    brief = ScopeBrief(
        archetype="explainer", topic="t", budget=get_agent_settings().budget, concepts=("x",)
    )
    return Contract.freeze("c", brief, Rubric(), brief.budget)
