"""`make dream`: consolidate the dream inbox into long-term memory.

Prints what the dream changed as a tree; with `--show-memory`, the whole
memory tree afterwards as well.
"""

import argparse
import asyncio
from collections.abc import Sequence

from agent_platform.memory.consolidation import change_tree_lines
from customer_support_agent.admin.show_memory import memory_lines, print_lines
from customer_support_agent.graph.builder import build_dream_service


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--show-memory", action="store_true", help="also print the full memory tree"
    )
    args = parser.parse_args(argv)
    report = asyncio.run(build_dream_service().dream())
    print_lines(change_tree_lines(report))
    if args.show_memory:
        print_lines(["", *memory_lines()])


if __name__ == "__main__":
    main()
