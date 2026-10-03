"""Box-drawn tables in plain text, for records a person reads as a grid.

Plain lines, like `text_tree`: a reply is text, and the same lines print in
a terminal, draw in the TUI feed and copy out with Ctrl-O unchanged. Colour
is not here; the TUI colours cells by pattern (`tui.response_highlights`),
so the record itself stays plain.
"""

from collections.abc import Sequence
from typing import Literal

from agent_platform.config import get_settings

Align = Literal["left", "right"]


def box_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], align: Sequence[Align]
) -> list[str]:
    """Header, rule, rows, each cell padded to its column's widest value."""
    widths = [max(len(text) for text in column) for column in zip(headers, *rows, strict=True)]
    glyphs = get_settings().presentation.box
    return [
        _rule(widths, glyphs.top, glyphs.horizontal),
        _row(headers, widths, align, glyphs.vertical),
        _rule(widths, glyphs.middle, glyphs.horizontal),
        *(_row(cells, widths, align, glyphs.vertical) for cells in rows),
        _rule(widths, glyphs.bottom, glyphs.horizontal),
    ]


def _rule(widths: Sequence[int], joins: Sequence[str], horizontal: str) -> str:
    left, middle, right = joins
    pad = get_settings().presentation.cell_padding
    return left + middle.join(horizontal * (width + 2 * pad) for width in widths) + right


def _row(cells: Sequence[str], widths: Sequence[int], align: Sequence[Align], vertical: str) -> str:
    pad = " " * get_settings().presentation.cell_padding
    shown = [
        text.rjust(width) if side == "right" else text.ljust(width)
        for text, width, side in zip(cells, widths, align, strict=True)
    ]
    return vertical + vertical.join(f"{pad}{text}{pad}" for text in shown) + vertical


__all__ = ["Align", "box_table"]
