"""The research agent, presented to the operator shell.

Everything the shell must say in this agent's vocabulary is gathered
here, behind the platform's `AgentHost` protocol and its optional hooks:
a greeting, a side panel (the live coverage checklist, then the run's
budget meters) and a cancel key. The platform composes it from
`agent.host` in this agent's overlay and never imports it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import RenderableType
from rich.table import Table

from agentic_commons.commands.types import Command
from agentic_commons.config import FeatureSet, get_settings
from agentic_commons.logging import get_logger
from agentic_commons.observability import StageReporter
from research_agent.config import get_agent_settings
from research_agent.console.commands import (
    CancelCommand,
    ForwardedCommand,
    HistoryCommand,
    ShowCommand,
)
from research_agent.console.greeting import build_greeting
from research_agent.console.setup import missing_model_key, search_usage, search_warning

_STATUS_STYLE = {"covered": "green", "waived": "yellow", "open": "white"}


class ResearchHost:
    """The research agent, as the shell sees it."""

    def title(self) -> str:
        return get_agent_settings().title

    def placeholder(self) -> str:
        return get_agent_settings().placeholder

    def greeting(self) -> str:
        settings, agent = get_settings(), get_agent_settings()
        notes = [
            note
            for note in (search_usage(settings, agent), search_warning(settings, agent))
            if note is not None
        ]
        return build_greeting(settings, agent, notes)

    def prepare(self) -> None:
        """Stop on a missing live credential; record the fixture world if replay needs it."""
        settings, agent = get_settings(), get_agent_settings()
        problem = missing_model_key(settings, agent)
        if problem is not None:
            raise SystemExit(problem)
        if settings.web.mode == "replay" and agent.model_backend == "fixture":
            from research_agent.fixture_world.generate import ensure_recorded

            fixtures = ensure_recorded()
            get_logger("research_agent").info("Fixture world ready", path=str(fixtures.folder))

    def build(self, reporter: StageReporter, features: FeatureSet | None = None) -> Any:
        """Compose the agent, reporting its stages to this feed."""
        from research_agent.graph import build_agent

        self._agent = build_agent(reporter=reporter, features=features)
        return self._agent

    @property
    def agent(self) -> Any:
        return getattr(self, "_agent", None)

    def commands(self) -> list[Command]:
        agent = get_agent_settings()
        names, help_text, messages = agent.command_names, agent.tui.command_help, agent.messages
        forwarded = [
            ForwardedCommand(names.qa, help_text["qa"], f"/{names.qa} <question>", needs_args=True),
            ForwardedCommand(
                names.research, help_text["research"], f"/{names.research} [question]"
            ),
            ForwardedCommand(names.done, help_text["done"], f"/{names.done}"),
            ForwardedCommand(
                names.waive, help_text["waive"], f"/{names.waive} <id|all>", needs_args=True
            ),
            ForwardedCommand(
                names.rubric, help_text["rubric"], f"/{names.rubric} [add|drop|set …]"
            ),
            ForwardedCommand(names.approve, help_text["approve"], f"/{names.approve}"),
        ]
        return [
            *forwarded,
            CancelCommand(
                names.cancel,
                help_text["cancel"],
                lambda: self.agent,
                (messages.cancel_requested, messages.no_active_run),
            ),
            HistoryCommand(
                names.history,
                help_text["history"],
                lambda: self.agent,
                agent.archive.history_limit,
                agent.tui.history_headers,
                messages.history_empty,
            ),
            ShowCommand(names.show, help_text["show"], lambda: self.agent, messages.unknown_run),
        ]

    def stage_labels(self) -> Mapping[str, str]:
        return get_agent_settings().tui.stage_labels

    # -- optional hooks -------------------------------------------------------------------

    def cancel_key(self) -> str:
        return get_agent_settings().cancel.key

    def cancel(self) -> bool:
        return bool(self.agent is not None and self.agent.cancel())

    def side_panels(self, session_id: str) -> Sequence[RenderableType]:
        """The live checklist while scoping; budget meters while a run is going."""
        if self.agent is None:
            return []
        state = self.agent.panel_state(session_id)
        labels = get_agent_settings().tui.panel_labels
        panels: list[RenderableType] = [
            f"[bold]{get_agent_settings().tui.panel_title}[/bold] · {state['phase']}"
        ]
        if state.get("items"):
            panels.append(_checklist(state["items"], labels["checklist"]))
        if state.get("budget"):
            panels.append(_meters(state, labels))
        usage = search_usage(get_settings(), get_agent_settings())
        if usage is not None:
            panels.append(usage)
        return panels


def _checklist(items: list[tuple[str, str]], title: str) -> Table:
    table = Table(title=title, show_header=False, expand=True)
    for item_id, status in items:
        table.add_row(item_id, f"[{_STATUS_STYLE.get(status, 'white')}]{status}[/]")
    return table


def _meters(state: dict[str, Any], labels: Mapping[str, str]) -> Table:
    table = Table(title=labels["run"], show_header=False, expand=True)
    for name, (spent, cap) in state["budget"].items():
        table.add_row(name, f"{spent:g} / {cap:g}")
    table.add_row(labels["ledger"], str(state.get("ledger", 0)))
    table.add_row(labels["rejected"], str(state.get("rejected", 0)))
    return table


__all__ = ["ResearchHost"]
