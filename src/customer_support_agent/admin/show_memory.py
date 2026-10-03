"""`make show-memory`: everything long-term memory holds, as a tree.

Read-only. The same tree `/memory` draws in the TUI, grouped by customer
and session, with generated ids left out.
"""

import argparse
from collections.abc import Sequence

from rich.console import Console

from agent_platform.config import get_settings
from agent_platform.memory.inspect import snapshot_tree, take_snapshot
from agent_platform.presentation.text_tree import tree_lines
from customer_support_agent.admin._store import agent_store


def memory_lines(customer: str | None = None, events: int | None = None) -> list[str]:
    """The memory tree for this agent, optionally narrowed to one customer."""
    settings = get_settings()
    snapshot = take_snapshot(agent_store(), settings.memory.episodic.resolved_path, customer)
    per_session = events or settings.commands.memory_tree_events_per_session
    return tree_lines(snapshot_tree(snapshot, per_session, settings.commands.tree_preview_chars))


def print_lines(lines: list[str]) -> None:
    """Print as-is: no markup, no wrapping mid-branch."""
    console = Console(highlight=False)
    for line in lines:
        console.print(line, markup=False, soft_wrap=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--customer", help="only this customer's memory")
    parser.add_argument("--events", type=int, help="events listed per session")
    args = parser.parse_args(argv)
    print_lines(memory_lines(args.customer, args.events))


if __name__ == "__main__":
    main()
