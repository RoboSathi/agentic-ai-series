"""The research agent, presented to the operator shell.

Everything the shell must say in this agent's vocabulary is gathered
here, behind the platform's `AgentHost` protocol and its optional hooks:
a greeting, a status block (coverage and evidence, expanding to the full
checklist and the budget meters) and a cancel key. The platform composes it from
`agent.host` in this agent's overlay and never imports it.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rich.markup import escape

from agent_platform.commands.types import Command
from agent_platform.config import FeatureSet, get_settings
from agent_platform.host import HostAction
from agent_platform.observability import StageReporter
from research_agent.config import get_agent_settings
from research_agent.console.commands import (
    CancelCommand,
    ForwardedCommand,
    HistoryCommand,
    ShowCommand,
)
from research_agent.console.greeting import build_greeting
from research_agent.console.setup import missing_model_key, search_warning
from research_agent.presentation.narration import say, title_of


class ResearchHost:
    """The research agent, as the shell sees it."""

    def title(self) -> str:
        return get_agent_settings().title

    def placeholder(self) -> str:
        return get_agent_settings().placeholder

    def greeting(self) -> str:
        settings, agent = get_settings(), get_agent_settings()
        warning = search_warning(settings, agent)
        notes = [warning] if warning is not None else []
        return build_greeting(agent, notes)

    def prepare(self) -> None:
        """Stop on a missing live credential; warm the local models."""
        settings, agent = get_settings(), get_agent_settings()
        problem = missing_model_key(settings, agent)
        if problem is not None:
            raise SystemExit(problem)
        self._preload_local_models(agent.audit.entailment_model == "platform")

    @staticmethod
    def _preload_local_models(nli: bool) -> None:
        """Load the local models now, before the TUI's event loop owns the process descriptors.

        Loaded lazily inside the TUI, the NLI model fails on macOS (`bad value(s) in
        fds_to_keep`) and every claim after that is judged by the lexical fallback. A model that
        cannot load still does not stop the console: the checker logs why and falls back.
        """
        from agent_platform.evals.entailment import preload_entailment_model
        from research_agent.graph.builder import intent_embedder

        if nli:
            preload_entailment_model()
        embedder = intent_embedder()
        if hasattr(embedder, "ensure_loaded_sync"):
            embedder.ensure_loaded_sync()

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

    def actions(self, session_id: str) -> Sequence[HostAction]:
        """The buttons for where the session stands: finish, approve or cancel."""
        if self.agent is None:
            return []
        settings = get_agent_settings()
        phase = self.agent.panel_state(session_id)["phase"]
        names = settings.command_names
        return [
            HostAction(
                spec.label,
                " ".join(
                    part for part in (f"/{getattr(names, spec.command)}", spec.argument) if part
                ),
                spec.during_turn,
            )
            for spec in settings.tui.actions.get(phase, [])
        ]

    def status_lines(self, session_id: str) -> tuple[Sequence[str], Sequence[str]]:
        """Nothing always shown: phase, coverage, evidence, budget and the checklist expand.

        The status line keeps one line (the current stage); everything about the run in
        progress sits behind its marker, so the feed above it keeps the room.
        """
        if self.agent is None:
            return [], []
        state = self.agent.panel_state(session_id)
        if not state.get("items") and not state.get("budget"):
            return [], []  # nothing in progress yet: no phase to report
        styles = get_agent_settings().tui
        phase = state["phase"]
        detail = [_styled(say("status.phase", phase=phase), styles.phase_colors.get(phase, "bold"))]
        items: list[tuple[str, str]] = state.get("items") or []
        if items:
            detail.append(_styled(_coverage_line(items), styles.status_line_styles["coverage"]))
        budget = state.get("budget")
        if budget:
            evidence = say(
                "status.evidence", n=state.get("ledger", 0), rejected=state.get("rejected", 0)
            )
            detail.append(_styled(evidence, styles.status_line_styles["evidence"]))
            meters = [_meter(name, spent, cap) for name, (spent, cap) in budget.items()]
            budget_line = say("status.budget", meters=" · ".join(meters))
            detail.append(_styled(budget_line, styles.status_line_styles["budget"]))
        if items:
            detail += _checklist_lines(items)
        return [], detail


def _styled(text: str, style: str) -> str:
    """`text` as Rich markup in `style` (plain when there is no style)."""
    return f"[{style}]{escape(text)}[/]" if style else escape(text)


def _coverage_line(items: list[tuple[str, str]]) -> str:
    counts = Counter(status for _, status in items)
    return say(
        "status.coverage", covered=counts["covered"], total=len(items), waived=counts["waived"]
    )


def _checklist_lines(items: list[tuple[str, str]]) -> list[str]:
    lines = [f"[bold]{say('status.detail.checklist')}[/bold]"]
    for item_id, status in items:
        style = get_agent_settings().tui.checklist_styles.get(status, "")
        lines.append(f"  {_styled(f'{status:<8}', style)} {escape(title_of(item_id))}")
    return lines


def _meter(name: str, spent: float, cap: float) -> str:
    return say("status.meter", name=title_of(name), spent=f"{spent:g}", cap=f"{cap:g}")


__all__ = ["ResearchHost"]
