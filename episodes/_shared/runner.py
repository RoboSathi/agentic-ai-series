"""Running one episode: the real agent, configured down (LLD §22).

An episode is not a different program from the product. It is the same
`build_agent()` the capstone runs and the same terminal app `make agent`
opens, with a `config.yaml` next to it that decides which subsystems are
live — which is the series' whole claim about harnesses, demonstrated
rather than described. Turn memory off in `e30/config.yaml` and the
memory steps disappear from the trace; nothing else changes.

Two entry points while the series converts:

`run_episode` — the current shape. Opens the TUI under the episode's
config, titled with the episode and its principle, and hands you the
prompt. You drive it.

`run_scripted_episode` — the shape being retired, where an episode
declared its conversation in config and a `flow` drove it through a
plain console. Still here for the episodes that have not moved yet.
"""

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent_platform.commands import Command
from agent_platform.config import reload_settings, use_overlay
from agent_platform.tui.app import AgenticTuiApp
from customer_support_agent.graph import AgentState, build_agent
from customer_support_agent.graph.agent import CustomerSupportAgent
from episodes._shared.settings import EpisodeSettings

# The episode's own config, found by convention beside its `app.py`. A
# name rather than a setting: an episode whose config had to be
# registered somewhere central would be an episode you could forget to
# register, and the failure would look like the harness misbehaving.
CONFIG_FILENAME = "config.yaml"

# Anything an episode must put in place before it can make its point —
# an order at an amount the demo data does not otherwise contain, say.
# Run after the config is live, so it sees the episode's database.
Setup = Callable[[], None]


def load_episode(episode_dir: Path) -> EpisodeSettings:
    """Apply an episode's configuration and read back what it declares.

    Separated from running it because the order matters and is easy to
    get wrong: `build_agent()` reads config to decide what to wire, so
    an overlay applied afterwards produces the fully-featured agent and
    an episode config that changed nothing.
    """
    use_overlay(episode_dir / CONFIG_FILENAME)
    reload_settings()
    return EpisodeSettings()  # type: ignore[call-arg]


def run_episode(
    episode_dir: Path,
    setup: Setup | None = None,
    commands: Sequence[Command] = (),
) -> None:
    """Open the terminal app on one episode's configuration."""
    episode = load_episode(episode_dir)
    if setup is not None:
        setup()
    try:
        AgenticTuiApp(
            lambda reporter: build_agent(reporter=reporter),
            title=f"{episode_dir.name.upper()} — {episode.title}",
            subtitle=episode.principle,
            commands=commands,
        ).run()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass


class EpisodeRun:
    """One scripted episode in progress: the agent, its script, the screen.

    Part of the shape being retired. Handed to an episode's `flow` so the
    episode writes what it is demonstrating and nothing about how a turn
    is rendered or which session it belongs to.
    """

    def __init__(
        self,
        agent: CustomerSupportAgent,
        settings: EpisodeSettings,
        session_id: str,
        console: Console,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.session_id = session_id
        self._console = console

    async def ask(self, turn: str) -> AgentState:
        """Put one customer message to the agent and show what came back."""
        self._console.print()
        self._console.print(Text(f"> {turn}", style="bold white"))
        state = await self.agent.handle(turn, session_id=self.session_id)
        self._console.print(Text(state.response, style="bright_green"))
        return state

    async def converse(self) -> list[AgentState]:
        """Run every turn the episode's config declares, in order."""
        return [await self.ask(turn) for turn in self.settings.turns]

    def note(self, line: str) -> None:
        """An aside from the episode's author, set apart from the agent."""
        self._console.print(Text(f"  {line}", style="dim italic"))


# What a scripted episode does, given a run in progress.
Flow = Callable[[EpisodeRun], Awaitable[None]]


def run_scripted_episode(episode_dir: Path, flow: Flow) -> None:
    """Run one episode's scripted conversation on a plain console."""
    episode = load_episode(episode_dir)

    console = Console()
    _banner(console, episode_dir.name, episode)

    asyncio.run(_drive(episode, episode_dir.name, console, flow))
    console.print()


async def _drive(episode: EpisodeSettings, session_id: str, console: Console, flow: Flow) -> None:
    """Build the agent this episode configured, and hand it to the flow."""
    await flow(EpisodeRun(build_agent(), episode, session_id, console))


def _banner(console: Console, name: str, episode: EpisodeSettings) -> None:
    """What this episode is, and the one thing it has to earn."""
    console.print()
    console.print(
        Panel(
            f"[bold]{episode.title}[/bold]\n[dim]{episode.principle}[/dim]",
            title=f"[bright_cyan]{name.upper()}[/bright_cyan]",
            border_style="bright_cyan",
        )
    )
