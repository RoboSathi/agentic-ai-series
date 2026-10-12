"""`make dream|show-memory|clean-memory|memory-reindex <agent>` — memory, from the terminal.

One entry point for every agent: the agent is chosen by its config layer
(`AGENTIC_OVERLAYS`, which the Makefile sets from the agent named), so
nothing here names one. The trees are the same lines `/dream` and
`/memory` draw in the console.
"""

import argparse
import asyncio
from collections.abc import Sequence

from rich.console import Console

from agent_platform.config import get_settings
from agent_platform.interface.commands.types import CommandResult
from agent_platform.interface.host.memory import agent_store, build_dream
from agent_platform.interface.presentation.terminal import print_result
from agent_platform.interface.presentation.text_tree import TreeNode, tree_lines
from agent_platform.memory import SemanticStore
from agent_platform.memory.dream import change_tree_lines
from agent_platform.memory.inspect import snapshot_tables, take_snapshot
from agent_platform.memory.reset import clear_long_term_memory

# What a reset deliberately leaves alone, said out loud so nobody wonders.
_KEPT = "Kept: conversation checkpoints, business records, reviews, audit, policy, knowledge index"
_RESTART = "Restart a running console: its working memory lives in the process."


def memory_result(subject: str | None = None, events: int | None = None) -> CommandResult:
    """Everything this agent's long-term memory holds, one table per store, the inbox last."""
    commands = get_settings().commands
    snapshot = take_snapshot(subject)
    per_session = events or commands.memory_tree_events_per_session
    result = CommandResult(title=commands.memory_tables.title.format(agent=snapshot.agent_id))
    result.line(snapshot.root)
    for table in snapshot_tables(snapshot, per_session):
        result.table(table)
    return result.line(commands.memory_tables.queue_note)


def dream_lines() -> list[str]:
    """Run a dream; what it changed."""
    return change_tree_lines(asyncio.run(build_dream().dream()))


def clean_lines() -> list[str]:
    """Empty long-term memory; what each store lost."""
    store = agent_store()
    removed = clear_long_term_memory(store)
    root = TreeNode(
        f"Cleared long-term memory · {store.agent_id} · {get_settings().memory.files.resolved_root}"
    )
    for label, count in removed.items():
        root.add(f"{label}: {count} removed")
    return [*tree_lines(root), _KEPT, _RESTART]


def reindex_lines() -> list[str]:
    """Rebuild the derived search index from the semantic records."""
    store = SemanticStore()
    return [f"Reindexed semantic memory · {store.reindex()} facts · {store.root}"]


def print_lines(lines: list[str]) -> None:
    """Print as-is: no markup, no wrapping mid-branch."""
    console = Console(highlight=False)
    for line in lines:
        console.print(line, markup=False, soft_wrap=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    actions = parser.add_subparsers(dest="action", required=True)
    dream = actions.add_parser("dream", help="consolidate pending dream events")
    dream.add_argument("--show-memory", action="store_true", help="also print the memory tree")
    show = actions.add_parser("show", help="print everything long-term memory holds")
    show.add_argument("--subject", help="only this subject's memory")
    show.add_argument("--events", type=int, help="events listed per session")
    actions.add_parser("clean", help="empty long-term memory")
    actions.add_parser("reindex", help="rebuild the derived search index")
    args = parser.parse_args(argv)
    if args.action == "dream":
        print_lines(dream_lines())
        if args.show_memory:
            print_result(memory_result())
    elif args.action == "show":
        print_result(memory_result(args.subject, args.events))
    elif args.action == "clean":
        print_lines(clean_lines())
    else:
        print_lines(reindex_lines())


if __name__ == "__main__":
    main()
