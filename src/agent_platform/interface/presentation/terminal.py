"""Printing a command's result in a terminal, exactly as the console draws it.

`make queue`, `make show-memory` and the other terminal doors call the
same command the console does and print its result through the same
`CommandView`, so a table looks the same — headings, grid, state colours —
wherever it is read.
"""

from rich.console import Console

from agent_platform.interface.commands.types import CommandResult
from agent_platform.interface.presentation.command_view import CommandView


def print_result(result: CommandResult, console: Console | None = None) -> None:
    """Every line of a command's result, styled, without wrapping a table mid-row."""
    out = console or Console(highlight=False)
    for line in CommandView().render(result):
        out.print(line, soft_wrap=True)


__all__ = ["print_result"]
