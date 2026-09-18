"""Slash-command contracts (LLD §20).

A slash command is an operator asking the harness about itself. It never
reaches a model and never touches the agent's conversation: `/audit` is a
query against the audit ledger, not a question the agent answers.

Keeping them separate matters for two reasons. An operator inspecting the
system should not consume the customer's token budget or appear in the
customer's transcript; and a command that went through the model could be
wrong about what the system contains, which defeats the point of asking.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ConsoleEffect(BaseModel):
    """Something only the console can do, asked for by a command.

    A command has no handle on the widget, and giving it one would make
    every command a UI component. It states the effect; the console
    applies it. `/clear` and `/debug` are the two that need this.
    """

    clear_transcript: bool = False
    # A logging level name, or "off". Empty means "leave the trace
    # stream alone" — which is what every other command wants.
    trace_level: str = ""


class CommandRow(BaseModel):
    """One row of tabular output."""

    cells: list[str]
    style: str = ""


class CommandResult(BaseModel):
    """What a command produced.

    Either a table or a list of lines — both, when a summary needs a
    heading. Commands return data rather than printing, so the same
    command works in the TUI, in a test, and eventually over an API.
    """

    title: str = ""
    lines: list[str] = Field(default_factory=list)
    headers: list[str] = Field(default_factory=list)
    rows: list[CommandRow] = Field(default_factory=list)
    ok: bool = True
    effect: ConsoleEffect = Field(default_factory=ConsoleEffect)

    @property
    def empty(self) -> bool:
        return not self.lines and not self.rows

    def line(self, text: str) -> "CommandResult":
        """Append one line, fluently."""
        self.lines.append(text)
        return self

    def row(self, *cells: str, style: str = "") -> "CommandResult":
        """Append one table row, fluently."""
        self.rows.append(CommandRow(cells=[str(cell) for cell in cells], style=style))
        return self


@runtime_checkable
class Command(Protocol):
    """One slash command.

    `name` is what the operator types after the slash; `summary` is what
    `/help` shows. Both are attributes rather than methods so the
    registry can list commands without running them.
    """

    name: str
    summary: str
    usage: str

    async def run(self, args: list[str]) -> CommandResult: ...


class CommandError(Exception):
    """Raised when a command cannot do what was asked.

    Distinct from a command that ran and found nothing: "no audit entries
    yet" is a result, "that is not a section of the config" is an error.
    """
