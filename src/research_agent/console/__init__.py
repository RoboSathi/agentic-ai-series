"""The research agent, presented to the operator shell.

Everything the shell must say in this agent's vocabulary is gathered
here, behind the platform's `AgentHost` protocol and its optional hooks:
a greeting, a status block (coverage and evidence, expanding to the full
checklist and the budget meters) and a cancel key. The platform composes it from
`agent.host` in this agent's overlay and never imports it.
"""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent_platform.commands.types import Command
from agent_platform.config import FeatureSet, get_settings
from agent_platform.durability.controls import run_actions
from agent_platform.host import HostAction
from agent_platform.logging import get_logger
from agent_platform.memory import MemoryProfile
from agent_platform.observability import StageReporter
from research_agent.config import get_agent_settings
from research_agent.console.commands import (
    CancelCommand,
    ForwardedCommand,
)
from research_agent.console.greeting import build_greeting
from research_agent.console.setup import missing_model_key, search_warning


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
        """Load the local models now, side by side, before the TUI's event loop owns the
        process descriptors.

        Loaded lazily inside the TUI, a local model fails on macOS (`bad value(s) in
        fds_to_keep`): the NLI model would leave every claim to the lexical fallback. The NLI
        model, the claim confirmer (when the agent turns it on) and the research-kind examples
        (the e5 embedder, and the examples it embeds once) load in parallel; all are resident
        for the life of the process. A model that cannot
        load does not stop the console: it is logged, and its caller falls back.
        """
        from agent_platform.hallucination_check.confirmer import preload_claim_confirmer
        from agent_platform.scoring.entailment import preload_entailment_model
        from research_agent.graph.builder import warm_kind_examples

        _start_presidio()
        agent = get_agent_settings()
        loads: list[Callable[[], None]] = [
            lambda: warm_kind_examples(agent),
            preload_claim_confirmer,
        ]
        if nli:
            loads.append(preload_entailment_model)
        with ThreadPoolExecutor(max_workers=len(loads)) as pool:
            for future in [pool.submit(load) for load in loads]:
                try:
                    future.result()
                except Exception as exc:
                    get_logger("research_agent").warning(
                        "A local model did not load; its caller will fall back", error=str(exc)
                    )

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
            ForwardedCommand(
                names.research, help_text["research"], f"/{names.research} [question]"
            ),
            ForwardedCommand(names.done, help_text["done"], f"/{names.done}"),
            ForwardedCommand(
                names.waive, help_text["waive"], f"/{names.waive} <id|all>", needs_args=True
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
        ]

    def stage_labels(self) -> Mapping[str, str]:
        return get_agent_settings().tui.stage_labels

    def memory_profile(self) -> MemoryProfile:
        """The default memory profile, reading research runs their own way (ra-0023)."""
        from research_agent.memory import research_memory_profile

        return research_memory_profile()

    # -- optional hooks -------------------------------------------------------------------

    def cancel_key(self) -> str:
        return get_agent_settings().cancel.key

    def cancel(self) -> bool:
        return bool(self.agent is not None and self.agent.cancel())

    def actions(self, session_id: str) -> Sequence[HostAction]:
        """The buttons for where the session stands: finish, approve or cancel."""
        if self.agent is None:
            return []
        # Stop while a run is in flight; Resume or Abort once one stopped: the platform's buttons.
        controls = run_actions(self.agent, session_id)
        if controls:
            return controls
        settings = get_agent_settings()
        phase = self.agent.phase(session_id)
        names = settings.command_names
        return [
            HostAction(
                spec.label,
                " ".join(
                    part for part in (f"/{getattr(names, spec.command)}", spec.argument) if part
                ),
                spec.during_turn,
                spec.variant,
            )
            for spec in settings.tui.actions.get(phase, [])
        ]

    def status_lines(self, session_id: str) -> tuple[Sequence[str], Sequence[str]]:
        """Nothing beneath the status line: the cards above already show the run's scope,
        evidence and budget (LOOP CONTROL), so a second copy behind a marker said nothing new."""
        return [], []


def _start_presidio() -> None:
    """The PII analyzer loads on its own thread, not waited for: the console opens at once."""
    if get_settings().security.pii.engine == "presidio":
        from agent_platform.security.pii.presidio import preload_presidio_in_background

        preload_presidio_in_background()


__all__ = ["ResearchHost"]
