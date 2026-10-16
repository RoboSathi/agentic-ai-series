"""Drawing a `CommandResult` into the feed (LLD §20).

Commands return data, not pixels — the same `/audit` result has to work
in the TUI, in a test, and eventually over an API. This is the one place
that knows what it looks like on a terminal.

A plain `rows` result is an aligned table; a `CommandTable` is a box
table coloured cell by cell, the way an agent's order table is. Both are
`Text`, so Ctrl-O copies them out as plain characters, and a terminal
command (`make queue`, `make show-memory`) prints the same `Text`.
"""

from collections.abc import Callable

from rich.text import Text

from agent_platform.config import get_settings
from agent_platform.interface.commands import CommandResult, CommandTable
from agent_platform.interface.presentation.text_table import column_widths, rule


class CommandView:
    """Turns one command result into styled feed lines."""

    def __init__(self) -> None:
        self._app_config = get_settings()
        self._style = get_settings().tui.command_output

    def render(self, result: CommandResult) -> list[Text]:
        """Title, table, then notes — each already styled."""
        lines: list[Text] = []
        if result.title:
            style = self._style.error_style if not result.ok else self._style.title_style
            lines.append(Text.styled(result.title, style))
        lines.extend(self._table(result))
        for table in result.tables:
            lines.extend(self.box(table))
        body_style = self._style.error_style if not result.ok else self._style.note_style
        lines.extend(Text.styled(line, body_style) for line in result.lines)
        return lines

    def box(self, table: CommandTable) -> list[Text]:
        """One titled box table, coloured as an order table is (see `command_output`)."""
        style = self._style
        lines = [Text.styled(table.title, style.table_title_style)] if table.title else []
        if not table.rows:
            return [*lines, Text.styled(table.empty, style.empty_style)] if table.empty else lines
        rows = [[self._fit(cell) for cell in row] for row in table.rows]
        headers = [self._fit(cell) for cell in table.headers]
        widths = column_widths(headers, rows)
        glyphs = self._app_config.presentation.box
        lines.append(Text.styled(rule(widths, glyphs.top), style.grid_style))
        lines.append(self._box_row(headers, widths, lambda _i, _c: style.table_header_style))
        lines.append(Text.styled(rule(widths, glyphs.middle), style.grid_style))
        lines.extend(self._box_row(row, widths, self._cell_style) for row in rows)
        lines.append(Text.styled(rule(widths, glyphs.bottom), style.grid_style))
        return lines

    def _box_row(
        self, cells: list[str], widths: list[int], style_of: Callable[[int, str], str]
    ) -> Text:
        presentation = self._app_config.presentation
        vertical, pad = presentation.box.vertical, " " * presentation.cell_padding
        text = Text(vertical, style=self._style.grid_style)
        for index, (cell, width) in enumerate(zip(cells, widths, strict=True)):
            text.append(pad)
            text.append(cell.ljust(width), style=style_of(index, cell))
            text.append(pad)
            text.append(vertical, style=self._style.grid_style)
        return text

    def _cell_style(self, index: int, cell: str) -> str:
        """A state word in its state's colour; the id column bold; anything else plain."""
        state = self._style.state_styles.get(cell.strip().casefold())
        if state is not None:
            return state
        return self._style.id_style if index == 0 else self._style.row_style

    def _fit(self, value: str) -> str:
        """One cell on one line, cut to the widest a column may be."""
        flat = " ".join(str(value).split())
        width = self._style.max_column_width
        return flat if len(flat) <= width else flat[: width - 1] + "…"

    def echo(self, line: str) -> Text:
        """The command the operator typed, echoed back into the feed."""
        return Text.styled(line, self._style.prompt_style)

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
            lines.append(Text.styled(header, self._style.header_style))
        for row in result.rows:
            body = gap.join(self._cell(c, w) for c, w in zip(row.cells, widths))
            lines.append(Text.styled(body, row.style or self._style.row_style))
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
