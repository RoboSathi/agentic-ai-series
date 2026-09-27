"""Drawing a `CommandResult` into the feed (LLD §20).

Commands return data, not pixels — the same `/audit` result has to work
in the TUI, in a test, and eventually over an API. This is the one place
that knows what it looks like on a terminal.

The rendering is a plain aligned table rather than a Rich `Table`,
because the feed is a single `Text` block that gets copied out verbatim
by Ctrl-O. A box-drawn table would copy as box-drawing characters and
paste into a ticket as noise.
"""

from rich.text import Text

from agentic_commons.commands import CommandResult
from agentic_commons.config import get_settings


class CommandView:
    """Turns one command result into styled feed lines."""

    def __init__(self) -> None:
        self._style = get_settings().tui.command_output

    def render(self, result: CommandResult) -> list[Text]:
        """Title, table, then notes — each already styled."""
        lines: list[Text] = []
        if result.title:
            style = self._style.error_style if not result.ok else self._style.title_style
            lines.append(Text(result.title, style=style))
        lines.extend(self._table(result))
        body_style = self._style.error_style if not result.ok else self._style.note_style
        lines.extend(Text(line, style=body_style) for line in result.lines)
        return lines

    def echo(self, line: str) -> Text:
        """The command the operator typed, echoed back into the feed."""
        return Text(line, style=self._style.prompt_style)

    def _table(self, result: CommandResult) -> list[Text]:
        """The rows as aligned columns, with an underlined header.

        Widths come from the content so a table of short values does not
        sprawl, capped so one long cell cannot push every other column
        off the right edge.
        """
        if not result.rows:
            return []
        grid = [result.headers] if result.headers else []
        grid += [row.cells for row in result.rows]
        widths = self._widths(grid)
        gap = " " * self._style.column_gap
        lines: list[Text] = []
        if result.headers:
            header = gap.join(self._cell(h, w) for h, w in zip(result.headers, widths))
            lines.append(Text(header, style=self._style.header_style))
        for row in result.rows:
            body = gap.join(self._cell(c, w) for c, w in zip(row.cells, widths))
            lines.append(Text(body, style=row.style or self._style.row_style))
        return lines

    def _widths(self, grid: list[list[str]]) -> list[int]:
        """The printed width of each column, capped."""
        columns = max(len(row) for row in grid)
        return [
            min(
                self._style.max_column_width,
                max((len(row[index]) for row in grid if index < len(row)), default=0),
            )
            for index in range(columns)
        ]

    def _cell(self, value: str, width: int) -> str:
        """One cell, truncated to the column width and padded to it."""
        text = value if len(value) <= width else value[: width - 1] + "…"
        return text.ljust(width)
