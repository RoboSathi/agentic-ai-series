"""What the shell needs from whatever agent it is running (LLD §20).

The TUI, the console and the command registry are the platform's
operator surface. They have to show *something* — a title, a set of
slash commands, a label for each stage that scrolls past — and every one
of those is a property of the agent being operated, not of the platform
operating it.

The obvious shortcut is for the shell to import the agent and ask it
directly. That is what this file replaces. It made `agentic_commons`
unable to run a second agent without editing `agentic_commons`, which is
the one thing a platform may not require: the dependency has to point
from the agent to the platform, never back.

So the shell depends on this Protocol, and an agent supplies an object
that satisfies it. `agentic_commons` still knows nothing about orders,
refunds or customers — it knows there is a host, and that a host has a
name and some commands.

An agent's host lives in its own package (Customer Support's is
`customer_support_agent.console.CustomerSupportHost`) and is named in
configuration, so which agent this process operates is a deployment
decision rather than an import.
"""

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from rich.console import RenderableType

from agentic_commons.commands.types import Command
from agentic_commons.config import FeatureSet
from agentic_commons.observability import StageReporter

# `module.path:attribute` — the same shape as a console-script entry
# point, because that is what this is.
SPEC_SEPARATOR = ":"


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
class PanelHost(Protocol):
    """An agent with live state worth a side panel (a checklist, run meters).

    Called on the panel's refresh tick with the console's session; returns
    Rich renderables. Must be cheap and must not raise: it reads state the
    agent already holds, it never computes it.
    """

    def side_panels(self, session_id: str) -> Sequence[RenderableType]: ...


@runtime_checkable
class CancellableHost(Protocol):
    """An agent whose long-running turn can be cancelled by a key binding."""

    def cancel_key(self) -> str:
        """The key that cancels, e.g. `ctrl+x`."""
        ...

    def cancel(self) -> bool:
        """Ask the running turn to stop; False when nothing was running."""
        ...


def load_spec(spec: str) -> Any:
    """Load the object named by a `module.path:attribute` string.

    Called when the attribute is callable (a class or factory), so a spec
    can name either an instance or the thing that builds one. Raises
    rather than falling back: a misspelled path must not quietly load
    nothing.
    """
    module_path, separator, attribute = spec.partition(SPEC_SEPARATOR)
    if not separator or not attribute:
        raise ValueError(f"'{spec}' is not a 'module.path{SPEC_SEPARATOR}attribute' spec.")
    module = importlib.import_module(module_path)
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"'{module_path}' has no attribute '{attribute}'.") from exc
    return factory() if callable(factory) else factory


def resolve_host(spec: str) -> AgentHost:
    """Load the host named by a `module.path:attribute` string.

    Raises rather than falling back to a default. A shell that quietly
    started with no agent because a path was misspelled would look like
    a working console attached to nothing.
    """
    host = load_spec(spec)
    if not isinstance(host, AgentHost):
        raise TypeError(f"'{spec}' does not satisfy the AgentHost protocol.")
    return host


__all__ = [
    "SPEC_SEPARATOR",
    "AgentHost",
    "CancellableHost",
    "GreetingHost",
    "PanelHost",
    "load_spec",
    "resolve_host",
]
