"""What the shell needs from whatever agent it is running (LLD §20).

The TUI, the console and the command registry are the platform's
operator surface. They have to show *something* — a title, a set of
slash commands, a label for each stage that scrolls past — and every one
of those is a property of the agent being operated, not of the platform
operating it.

The obvious shortcut is for the shell to import the agent and ask it
directly. That is what this file replaces. It made `agent_platform`
unable to run a second agent without editing `agent_platform`, which is
the one thing a platform may not require: the dependency has to point
from the agent to the platform, never back.

So the shell depends on this Protocol, and an agent supplies an object
that satisfies it. `agent_platform` still knows nothing about orders,
refunds or customers — it knows there is a host, and that a host has a
name and some commands.

An agent's host lives in its own package (Customer Support's is
`customer_support_agent.console.CustomerSupportHost`) and is named in
configuration, so which agent this process operates is a deployment
decision rather than an import.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_platform.commands.types import Command
from agent_platform.config import FeatureSet
from agent_platform.observability import StageReporter


@runtime_checkable
class AgentHost(Protocol):
    """One agent, presented to the operator shell.

    Deliberately small. Everything here is something the *shell* has to
    render or dispatch; anything the agent merely needs internally stays
    inside the agent, where the shell has no business knowing about it.
    """

    def title(self) -> str:
        """What this agent is called, for the window title."""
        ...

    def placeholder(self) -> str:
        """The prompt shown in an empty input box."""
        ...

    def prepare(self) -> None:
        """Do any slow start-up work before the event loop opens.

        Loading a local classifier after the TUI has opened its
        descriptors is how the macOS `fds_to_keep` failure appears, so
        an agent with native dependencies warms them here rather than on
        the first turn.
        """
        ...

    def build(self, reporter: StageReporter, features: FeatureSet | None = None) -> Any:
        """Compose the agent, reporting its stages to this feed.

        `features` is how `/harness` rebuilds: the composition root is
        handed a posture rather than reading one, so two agents in one
        process can hold different ones.
        """
        ...

    def commands(self) -> list[Command]:
        """This agent's own slash commands, added to the platform's."""
        ...

    def stage_labels(self) -> Mapping[str, str]:
        """Display names for the stage keys only this agent emits."""
        ...


# -- optional hooks --------------------------------------------------------------
# An agent may offer any of these; the shell asks with `isinstance` and
# behaves exactly as before when they are absent, so an agent that has
# never heard of them (Customer Support) is unaffected.


@runtime_checkable
class GreetingHost(Protocol):
    """An agent that says its own first words when the console opens."""

    def greeting(self) -> str: ...


@runtime_checkable
class StatusHost(Protocol):
    """An agent with live state worth a status block under the feed.

    Called on the block's refresh tick with the console's session; returns
    the lines always shown, then the lines shown when the operator expands
    the block. Lines are Rich markup. Must be cheap and must not raise: it
    reads state the agent already holds, it never computes it.
    """

    def status_lines(self, session_id: str) -> tuple[Sequence[str], Sequence[str]]: ...


@dataclass(frozen=True)
class HostAction:
    """One button the console offers: what it says, and the command it runs when pressed."""

    label: str
    command: str
    # Whether it may be pressed while a turn is running (a cancel can; a "finish" cannot).
    during_turn: bool = False
    # How it is drawn: Textual's button variants (`default`, `primary`, `success`,
    # `warning`, `error`), so a finishing step reads differently from a stop.
    variant: str = "default"


@runtime_checkable
class ActionHost(Protocol):
    """An agent with next steps worth a button beside the slash command (done, approve, cancel).

    Called on the status block's refresh tick with the console's session. A press runs the
    command exactly as if it had been typed. Must be cheap and must not raise.
    """

    def actions(self, session_id: str) -> Sequence[HostAction]: ...


@runtime_checkable
class SubjectHost(Protocol):
    """An agent that can say who has proved who they are in a console session.

    The answer is the opaque `subject_id` the agent's records carry, or None
    while nobody is verified. Console commands that read people's records
    show only that subject's (`ConsoleScope`).
    """

    def verified_subject(self, session_id: str) -> str | None: ...


@runtime_checkable
class CancellableHost(Protocol):
    """An agent whose long-running turn can be cancelled by a key binding."""

    def cancel_key(self) -> str:
        """The key that cancels, e.g. `ctrl+x`."""
        ...

    def cancel(self) -> bool:
        """Ask the running turn to stop; False when nothing was running."""
        ...


__all__ = [
    "ActionHost",
    "AgentHost",
    "CancellableHost",
    "GreetingHost",
    "HostAction",
    "StatusHost",
    "SubjectHost",
]
